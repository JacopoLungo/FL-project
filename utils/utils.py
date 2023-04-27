import torch.nn as nn


class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()

def unNormalize(self, tensorImage, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for t, m, s in zip(tensorImage, mean, std):
            t.mul_(s).add_(m)
        return tensorImage
