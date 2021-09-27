"""
    dataset object (data streamer) over image with corresponding boundinx box, with
    labelisation of objects
"""

from torch.utils.data import Dataset
import pandas as pd,numpy as np

from PIL import Image,ImageDraw
from pathlib import Path
dir_file = Path(__file__).parent

try:
    from .utils import bbox_mod
    from ..Data import data_manager
except:
    from Code.utils import bbox_mod
    from Data import data_manager


class DatasetRCNN(Dataset):
    def __init__(self,mode="train",input_transform=None,target_transform=None):
        """initisation of the dataset
            :mode : can be equal to train, val or test
        """

    ######## init options
        self.data_accessor = data_manager.DataAccessor(mode)
        self.groundTruth = self.read_detections_file()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def read_detections_file(self):
        return self.data_accessor.read_csv_file()

    def get_pil_image_from_name(self,image_id):
        return self.data_accessor.get_pil_image_from_name(image_id)

    def __getitem__(self, idx):
        for i,(imageID, df) in enumerate(alg.groundTruth.groupby("ImageID")):
            if i == idx:
                break


        pil_img = self.get_pil_image_from_name(imageID)

        col,row = pil_img.size

        # el = self.groundTruth.iloc[idx]
        bboxes = []
        labels = []
        for row_df,el in df.iterrows():

            bbox = ((el.XMin*col,el.YMin*row),(el.XMax*col,el.YMax*row))
            label_name = data_manager.code_to_name_of_class[el.LabelName]

            bbox = bbox_mod.Bbox.from_extremes(*bbox)
            bboxes.append(bbox)
            labels.append(label_name)

        return pil_img,bboxes,labels

        #TODO : add transforms here

        # transforms from  image + bboxes to assocaitions (selective_search_bbox,bbox)
        # transforms from (selective_search_bbox,bbox) to offsets
        # transforms from label to one_hot_encoding


        ##########
        # return pil_img,label_name,bbox
        # inpt,outpt = pil_img,(label_name,bbox)
        # if self.input_transform:
        #     inpt = self.input_transform(inpt)
        # if self.target_transform:
        #     outpt = self.target_transform(outpt)
        #
        # return inpt,outpt



    def draw_bbox_on_image(self,idx):
        img, bboxes,labels = alg[idx]
        for bbox in bboxes:
            bbox.draw_on_image(img,False)
        img.show()


if __name__ == '__main__':
    alg = DatasetRCNN()
    # self = alg
    pass
    # r = self.csv_iterator


