from torch import nn
from torchvision import models
import numpy as np
import torch


class YoloModel(nn.Module):
    def __init__(self,nb_classes = 3,S=7,B=2):
        super().__init__()
        self.nb_classes = nb_classes
        self.S = S
        self.B = B

        self.shape_dst = 5*self.B+self.nb_classes,self.S,self.S


        self.vgg16_bn_feat = self.get_vgg16_bn_feaures()

        # we build a convolutionnal layer with a batch Norm, leakyRELU, and Dropout layer, on
        # top of the pretrained layer for classification (as recommended by YOLO)
        self.new_conv_layer = self.construct_conv_layer()

        self.drop_out = nn.Dropout(0.5)
        # we build then a fully connected layer, to outputs predictions
        self.fully_conn_layer = self.get_fully_connected_Layer()


    def get_vgg16_bn_feaures(self):
        """
        :return: feature part of the neural network vgg16 bn, preTrained on ImageNet
        """
        vgg16_bn_feat = models.vgg16_bn(pretrained=True).features

        for module in vgg16_bn_feat:
            if hasattr(module, "inplace"):
                setattr(module,"inplace",False)

        return vgg16_bn_feat

    def construct_conv_layer(self,in_ch= 512,out_ch=1024):
        res = nn.Sequential(nn.Conv2d(in_ch,out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.LeakyReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        return res

    def get_fully_connected_Layer(self):
        fully_conn_layer_1 = nn.Sequential(nn.Flatten(),
                           nn.Linear(1024 * self.S*self.S, 4096),
                           nn.LeakyReLU())
        fully_conn_layer_2 = nn.Sequential(nn.Linear(4096, np.prod(self.shape_dst)),
                           nn.LeakyReLU(),
                           nn.Unflatten(1, self.shape_dst))
        fully_conn_layer = nn.Sequential(fully_conn_layer_1,fully_conn_layer_2)
        return fully_conn_layer


    def forward(self,inpt):
        tmp = self.vgg16_bn_feat(inpt)
        tmp = self.drop_out(tmp)
        tmp = self.new_conv_layer(tmp)
        tmp = self.fully_conn_layer(tmp)

        out = tmp
        return out



if __name__ == '__main__':
    model = YoloModel(3)

    N = 10
    self = model
    inpt = torch.ones(N, 3, 448, 448)
    tmp = self.vgg16_bn_feat(inpt)
    tmp = self.drop_out(tmp)
    tmp = self.new_conv_layer(tmp)
    assert tmp.shape[1:] == torch.Size((1024 ,self.S,self.S))
    out = self.fully_conn_layer(tmp)

    assert out.shape[1:] == torch.Size((self.S,self.S,5*self.B + self.nb_classes))

    assert out.shape[0] == N