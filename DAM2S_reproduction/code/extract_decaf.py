"""4096-d DeCAF6 features (fc6 + ReLU of an ImageNet AlexNet) for all RGB images.

The paper uses Caffe's DeCAF6; we use torchvision's ImageNet AlexNet, whose fc6 layer
plays the same role.  Each image is warped to 256x256 and the central 224x224 crop is used.

Usage: python -I extract_decaf.py <or_split.json> <out.npz>
"""
import json
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T


class FC6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        self.features, self.avgpool = net.features, net.avgpool
        self.fc6 = torch.nn.Sequential(net.classifier[1], net.classifier[2])  # Linear(9216,4096)+ReLU

    def forward(self, x):
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.fc6(x)


def run(paths, model, dev, bs=128):
    tf = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
        with torch.no_grad():
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def main(split_json, out_npz):
    split = json.load(open(split_json))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FC6().eval().to(dev)
    xs = run([s["rgb"] for s in split["source"]], model, dev)
    xt = run([s["rgb"] for s in split["target"]], model, dev)
    np.savez_compressed(out_npz, Xs=xs, Xt=xt,
                        ys=np.array([s["label"] for s in split["source"]]),
                        yt=np.array([s["label"] for s in split["target"]]))
    print("DeCAF6:", xs.shape, xt.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
