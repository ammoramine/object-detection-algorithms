import torch
from torch import nn
from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class RCNN(nn.Module):
    def __init__(self,nb_classes = 3):
        """

        :param nb_classes: must include the background so nb_classes = effective_nb_classes + 1
        """
        super().__init__()
        self.vgg_classifier = self.get_vgg_classifier()
        self.nb_classes = nb_classes
        feature_dim = 25088
        # fixed using as input the number of dimensions of the output fo vgg features
        #TODO : computes it automatically !

        self.cls_score = nn.Linear(feature_dim, self.nb_classes)
        self.bbox = nn.Sequential(
              nn.Linear(feature_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 4),
              nn.Tanh(),
            )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()

    def get_vgg_classifier(self):
        vgg_classifier = models.vgg16(pretrained=True)
        vgg_classifier.classifier = nn.Sequential()
        for param in vgg_classifier.parameters():
            param.requires_grad = False
        vgg_classifier.eval().to(device)
        return vgg_classifier

    def forward(self, input):
        features = self.vgg_classifier(input)
        cls_score = self.cls_score(features)
        bbox = self.bbox(features)
        return cls_score, bbox

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from functools import partial

    from  Code.datasets.dataset_RCNN import DatasetRCNN
    train_ds = DatasetRCNN(mode="train")
    collate_fn = partial(train_ds.collate_fn,device=device)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn, drop_last=True,shuffle=True)

    alg = RCNN(3)

