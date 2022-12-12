import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_accuracy(logits,label):
    _, pred = torch.max(logits.data, 1)
    return ((label.data == pred).float().mean())
class ArcFace(nn.Module):
    def __init__(self, feature_size, class_num, s=64, m=0.1):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.randn(feature_size, class_num)).to(device)
        self.s = s
        self.m = m

    def forward(self, feature, label):
        w = F.normalize(self.weight, dim=0)
        feature_norm = F.normalize(feature, dim=1)
        cosa = torch.matmul(feature_norm, w) / self.s
        a = torch.acos(cosa)
        top = torch.exp(torch.cos(a + self.m) * self.s)
        _top = torch.exp(torch.cos(a) * self.s)
        bottom = torch.sum(_top, dim=1, keepdim=True)
        logits = (top / (bottom - _top + top)) + 1e-10
        accuracy = calculate_accuracy(logits,label)
        return logits,accuracy
