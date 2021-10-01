from torchmetrics import Metric
import torch

class RCNNMetric(Metric):
    def __init__(self,nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
        self.add_state("confusion_matrix",torch.zeros(nb_classes,nb_classes).long(),sum)

    def update(self,preds,targets):
        """
        :param preds: of shape (B,C) where C = self.nb_classes
        :param targets: of shape (B), containins ints on range({self.nb_classes})
        :return:
        """
        preds = torch.argmax(preds,axis=1)
        for pred,target in zip(preds,targets):
            self.confusion_matrix[pred,target] += 1


    def compute(self):
        return torch.sum(torch.diag(self.confusion_matrix))/torch.sum(self.confusion_matrix)
