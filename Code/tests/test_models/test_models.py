import pytest
import torch

from Code.models import yolo_v3model


def test_yolo_model():
    """
        assert some assumptions about Yolo Model, concerning shape of tensors along
        the network
    """
    model = yolo_v3model.YoloModel(3)
    N = 10
    inpt = torch.ones(N, 3, 448, 448)
    tmp = model.vgg16_bn_feat(inpt)
    tmp = model.drop_out(tmp)
    tmp = model.new_conv_layer(tmp)
    assert tmp.shape[1:] == torch.Size((1024 ,model.S,model.S))
    out = model.fully_conn_layer(tmp)

    assert out.shape[1:] == torch.Size((model.S,model.S,5*model.B + model.nb_classes))

    assert out.shape[0] == N