from torch import optim,nn
import torch
class RCNNLoss:
    def __init__(self):
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
        self.lmb = 10.0

    def loss_fn(self, pred, target):
        probs, _deltas = pred
        labels, deltas = target

        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != 0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]

        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss
