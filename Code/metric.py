from torchmetrics import Metric
import torch

class RCNNMetric(Metric):
    def __init__(self,nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
        self.add_state("confusion_matrix",torch.zeros(nb_classes,nb_classes).long(),sum)

    def update(self,preds,targets):
        for pred,target in (preds,targets):
            pred = torch.argmax(pred[0], axis=1)
            self.confusion_matrix[pred,target] += 1


    def compute(self):
        return torch.sum(torch.diag(self.confusion_matrix))/torch.sum(self.confusion_matrix)
