"""
    dataset object (data streamer) over image with corresponding boundinx box , and
    labelisation of objects, that outputs an iterable, that gives correspondance,
    between bbox (bounding boxes) , generated by the selective search algorithm,
    and the labelled bounding boxes from the csv
"""

from torch.utils.data import Dataset
import pandas as pd,numpy as np,cv2
from torchvision import transforms
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from PIL import Image,ImageDraw
from pathlib import Path
dir_file = Path(__file__).parent



try:
    from .. import utils
    from ..utils import bbox_mod
    from ...Data import data_manager
    from .. import custom_transforms
except:
    from Code import utils
    from Code.utils import bbox_mod
    from Data import data_manager
    from Code import custom_transforms


class DatasetRCNN(Dataset):
    def __init__(self,mode="train",with_final_transform = True):
        """initisation of the dataset
            :mode : can be equal to train, val or test
        """
        self.mode = mode
        self.path_csv = self.read_prepared_data_for_RCNN()
        self.data_accessor = data_manager.DataAccessor(mode)
        self.with_final_transform = with_final_transform



        self.df = utils.read_csv_and_eval(self.path_csv,collumns_eval=["p_bboxes",'gd_bboxes','labels','offsets'])

        self.all_labels = self.get_all_labels()

        self.final_RCNN_transform_inst = custom_transforms.FinalRCNNTransform(self.all_labels)

    def get_all_labels(self):
        all_labels = []
        [all_labels.extend(el.labels) for idx, el in self.df.iterrows()]

        all_labels = utils.remove_duplicate_of_iterable(all_labels)
        return all_labels


    def read_prepared_data_for_RCNN(self):
        return data_manager.get_path_detections_csv_for_RCNN_dataset(self.mode)

    def get_pil_image_from_name(self,image_id):
        return self.data_accessor.get_pil_image_from_name(image_id)

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        imageID,p_bboxes,gd_bboxes,labels,offsets  = self.df.iloc[idx]

        pil_img = self.get_pil_image_from_name(imageID)

        el = pil_img,p_bboxes,gd_bboxes,labels,offsets
        if self.with_final_transform:
            crops,labels_as_targets,offsets =  self.final_RCNN_transform_inst(el)
            return crops,labels_as_targets,offsets
        return el







    def collate_fn(self, batch,device):
        """this is a function, that must be given as output to the DataLoader object """

        if not(self.with_final_transform):
            raise ValueError(f"collate_fn of the dataset {self} shouldn't"
                             f"be called, because the final transformation, wasn't done"
                             f"instantiate the dataset with the self.with_final_transform == True")

        crops,labels,deltas = [], [] , []

        for el in batch:
            crops_inst, labels_inst, offsets_inst = el
            crops.extend(crops_inst)
            labels.extend(labels_inst)
            deltas.extend(offsets_inst)

        crops = torch.cat(crops).to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return crops, labels, deltas



    def draw_bbox_on_image(self,idx):
        """function, to show input datat for particualr image"""
        if self.with_final_transform:
            raise ValueError(f"function shouldn't be called, because the transformation"
                             f"self.with_final_transform is set to True of the object"
                             f"{self}")
        pil_img,p_bboxes,gd_bboxes,labels,offsets = self[idx]
        for bbox,label in zip(p_bboxes,labels):
            if label != "Background":
                bbox.draw_on_image(pil_img,False)
        pil_img.show()

#TODO : to be tested

if __name__ == '__main__':
    from Code import custom_transforms
    from Code import utils


    train_ds = DatasetRCNN(mode="train")

    from torch.utils.data import DataLoader
    from functools import partial
    collate_fn = partial(train_ds.collate_fn,device=device)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn, drop_last=True,shuffle=True)
