import torch


def calc_loss(self, probs, _deltas, labels, deltas):
    detection_loss = self.cel(probs, labels)
    ixs, = torch.where(labels != 0)
    _deltas = _deltas[ixs]
    deltas = deltas[ixs]
    self.lmb = 10.0
    if len(ixs) > 0:
        regression_loss = self.sl1(_deltas, deltas)
        return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
    else:
        regression_loss = 0
        return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss