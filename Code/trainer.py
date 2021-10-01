import torch
try:
    from . import loss
    from .models import rcnn_model
except:
    from Code import loss
    from Code.models import rcnn_model

from torch import optim
from torch.utils.data import DataLoader
from functools import partial

from  Code.datasets.dataset_RCNN import DatasetRCNN
# loss_func = RCNNLoss().loss_fn
# metric = RCNNMetric(3)
# optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

class Trainer:
    def __init__(self,model,train_loader,val_loader,loss_func,metric,optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.metric = metric
        self.optimizer = optimizer


    def get_res_on_epoch(self,batch_loader,batch_processor):
        """
        :param batch_loader: an iterable, that outputs data at each iteration
        :param batch_processor: a functoin that takes as input the data given by the batch_loader
        anf outputs 3 loss values, and an accuravy value
        :return: mean value of the batch_processor, over all the iterations
        """
        self.metric.reset()
        loss,loc_loss,regr_loss,accs = 0,0,0,0
        for data in batch_loader:
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

            message = f"result for epoch :"
            if isinstance(epoch,int):
                message += f"{epoch}"
            for mode,res in zip(["training","validation"],[trn_out,val_out]):
                message += f"{mode} total loss is {res[0]}"
                message += f"{mode} loc loss is {res[1]}"
                message += f"{mode} regr loss is {res[2]}"
                message += f"{mode} accuracy loss is {res[3]}"
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = DatasetRCNN(mode="train")
    collate_fn = partial(train_ds.collate_fn,device=device)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn, drop_last=True,shuffle=True)

    val_ds = DatasetRCNN(mode="validation")
    collate_fn = partial(val_ds.collate_fn,device=device)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn, drop_last=True,shuffle=True)


    nb_classes = 3
    model = rcnn_model.RCNN(nb_classes)

    loss_func = loss.RCNNLoss()

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    from Code.metric import RCNNMetric

    metric = RCNNMetric(nb_classes)



    alg_trainer =  Trainer(model, train_loader, val_loader, loss_func, metric, optimizer)