import torch
try:
    from . import loss
    from .models import rcnn_model
except:
    from Code import loss
    from Code.models import rcnn_model

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from collections import namedtuple

from  Code.datasets.dataset_RCNN import DatasetRCNN

Results = namedtuple("Results",["loss", "loc_loss", "regr_loss","accuracy"])

def add_results_info_to(self, message=""):
    """show informations about selfs"""
    message += f" total loss is {self.loss}  "
    message += f" loc loss is {self.loc_loss}  "
    message += f" regr loss is {self.regr_loss}  "
    message += f" accuracy loss is {self.accuracy} \n "
    return message
Results.add_results_info_to = add_results_info_to

class Trainer:
    def __init__(self,model,train_loader,val_loader,loss_func,metric,optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.metric = metric
        self.optimizer = optimizer

        self.val_results = []
        self.train_results = []



    def get_res_on_epoch(self,batch_loader,batch_processor):
        """
        :param batch_loader: an iterable, that outputs data at each iteration
        :param batch_processor: a functoin that takes as input the data given by the batch_loader
        anf outputs 3 loss values, and an accuravy value
        :return: mean value of the batch_processor, over all the iterations
        """
        self.metric.reset()
        loss,loc_loss,regr_loss,accs = 0,0,0,0
        for data in tqdm(batch_loader):
            loss_tmp, loc_loss_tmp, regr_loss_tmp = batch_processor(data)
            loss += loss_tmp
            loc_loss += loc_loss_tmp
            regr_loss += regr_loss_tmp
        acc = self.metric.compute()
        out = loss, loc_loss, regr_loss
        out = [el/len(batch_loader) for el in out]
        out += [acc]
        return tuple(out)

    def iterate_over_epoch(self):
        """
        update parameters of models and print mean loss value, and accuracy over an epoch
        :param epoch: can be int or None,
        """
        trn_out = self.get_res_on_epoch(self.train_loader,self.train_batch)

        val_out = self.get_res_on_epoch(self.val_loader,self.validate_batch)
        return trn_out,val_out

    def iterate_over_multiple_epochs(self,nb_epochs):
        for epoch in range(nb_epochs):
            trn_out, val_out = self.iterate_over_epoch()

            res_val = Results(*val_out)
            res_trn = Results(*trn_out)
            self.val_results.append(res_val)
            self.train_results.append(res_trn)

            message = f" Results for epoch {epoch} : \n"

            for mode,res in zip(["training","validation"],[res_trn,res_val]):
                message += f"mode {mode} \n"
                message =res.add_results_info_to(message)
                print(message)

    def train_batch(self,data):
        self.model.train()
        self.optimizer.zero_grad()
        loss,loc_loss,regr_loss = self.compute_on_batch(data)
        loss.backward()
        self.optimizer.step()
        return loss.detach(), loc_loss, regr_loss

    @torch.no_grad()
    def validate_batch(self,data):
        self.model.eval()
        loss, loc_loss, regr_loss = self.compute_on_batch(data)
        return loss.detach(), loc_loss, regr_loss

    def compute_on_batch(self,data):
        """the total loss, the regression loss,  the localisation loss, and the accuracy"""
        crops, clss_target, deltas_target = data
        clss_pred,deltas_pred = self.model(crops)

        target = clss_target,deltas_target
        pred = clss_pred,deltas_pred
        loss, loc_loss, regr_loss = self.loss_func(pred, target)

        self.metric.update(clss_pred,clss_target)
        #TODO : include accuracy for bbox detection (look at YOLO)
        return loss,loc_loss,regr_loss


if __name__ == '__main__':

    device = "cpu"
    train_ds = DatasetRCNN(mode="train")
    val_ds = DatasetRCNN(mode="validation")

    # ds =

    from Code import utils
    trn_collate_fn = partial(train_ds.collate_fn,device=device)
    val_collate_fn = partial(val_ds.collate_fn,device=device)

    truncate = 3
    train_ds,val_ds = [ utils.truncate_dataset(ds,truncate) for ds in [train_ds,val_ds]]

    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=trn_collate_fn, drop_last=True,shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=val_collate_fn, drop_last=True,shuffle=True)


    nb_classes = 3
    model = rcnn_model.RCNN(nb_classes).to(device)

    loss_func = loss.RCNNLoss()

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    from Code.metric import RCNNMetric

    metric = RCNNMetric(nb_classes)



    args = dict()
    args["model"] = model
    args["train_loader"] = train_loader
    args["val_loader"] = val_loader
    args["loss_func"] = loss_func
    args["metric"] = metric
    args["optimizer"] = optimizer



    alg_trainer =  Trainer(**args)

    alg_trainer.iterate_over_multiple_epochs(1)