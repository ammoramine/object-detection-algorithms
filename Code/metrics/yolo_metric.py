from torchmetrics import Metric
import torch

class YOLOMetric(Metric):
    def __init__(self,nb_classes,S,min_iou,min_confidence):
        """

        :param nb_classes: used for the classificaiton (including background)
        :param minIOU: minimmum value of IOU, to consider the class correctly classified
        """
        super().__init__()
        self.nb_classes = nb_classes
        self.S = S
        self.min_iou = min_iou # if it is
        self.min_confidence = min_confidence
        import torch
        # self.add_state("localization_error",torch.zeros(self.nb_classes))
        # localization error per class
        self.add_state("classification",torch.Tensor(self.nb_classes+1,self.nb_classes))
        # at (i,j) the number of samples from class j predicted as i for i < nb_classes-1
        # with correct classification
        # at (nb_classes,j), the sample from class j is badly localized
        # self.add_state("",torch.Tensor(self.nb_classes,self.nb_classes))
    def update(self,preds,targets):
        """
        preds and target are supposed to be filtered
        :param preds: tensor of shape (N,S,S,5*B+nb_classses)
        :param targets: tensor of same shape as preds
        :return:
        """

        # we parse the tensor preds value to B tensor, one for each box
        # plus a tensor for classifiaction
        # we do the same from the targets, we compute IOU,
        # we create a tensor fo shape (B,B), we assignate -1 , for the
        # diagonal, and for all the target bbox for which the confidence
        # is below 0.5, we compute the IOU for the rest (bbox from
        # target and preds ,whose confidence is above self.min_confidence)
        # we associate the for each pred bbox, the one from the target
        # with the max iou
        # for each pred_bbox, if the iou, is below a certain value , we keep
        # it
        # preds and targets have for each cell multiple bboxes.
        # these multiple bboxes should be correctly associated,
        # we should thence write routines for that.


        preds = preds.reshape(preds.shape[0], -1, preds.shape[-1])
        targets = targets.reshape(targets.shape[0], -1, targets.shape[-1])
        for pred,target in zip(preds,targets):
            for pred_cell,target_cell in zip(pred,target):
                res = pred_cell.match_bbox(target_cell) #bbox_with_content_object
                self.update_classification(res)

    def compute(self):
        """
        :return: the mean accuracy, should returns the mean average precision instead
        #TODO : change with the mean average accuracy instead
        """
        mean_accuracy = torch.diag(self.classification).sum()/self.classification.sum()
        return mean_accuracy