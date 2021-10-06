from torch import optim,nn
import torch
class YOLOLoss:
    def __init__(self):
        # self.cel = nn.CrossEntropyLoss()
        # self.sl1 = nn.L1Loss()
        self.lbd_coord = 5.0
        self.lbd_no_obj = 0.5

    def __call__(self, pred, target):
        res = 0
        preds_bboxes,targets_bboxes = [[tmp[...,:5],tmp[...,5:10] ] for tmp in [pred,target]]
        onesij = []
        for pred_box,target_bbox in zip(preds_bboxes,targets_bboxes):
            Ci , Ci_tilde = pred_box[...,4],target_bbox[...,4]
            _,one_ij = Ci , Ci_tilde
            onesij.append(one_ij)

            pred_box_xy, target_bbox_xy = pred_box[...,:2], target_bbox[...,:2]
            res = torch.sum(  ((pred_box_xy - target_bbox_xy)**2).sum(axis=-1) * one_ij )

            pred_box_wh, target_bbox_wh = pred_box[...,2:4], target_bbox[...,2:4]
            res += torch.sum( ((torch.sqrt(pred_box_wh) - torch.sqrt(target_bbox_wh))**2).sum(axis=-1) * one_ij)

            res *= self.lbd_coord

            res += torch.sum(one_ij * (Ci- Ci_tilde)**2)
            res += self.lbd_no_obj * torch.sum( (1-one_ij) * (Ci- Ci_tilde)**2)

        preds_classification,target_classification = [tmp[...,10:] for tmp in [pred,target]]

        onesi = onesij.sum(axis=0)
        res += torch.sum(onesi * ((preds_classification - target_classification)**2).sum(axis=-1))

        return res