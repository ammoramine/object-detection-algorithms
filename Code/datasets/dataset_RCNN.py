# """
#     dataset object (data streamer) over image with corresponding boundinx box , and
#     labelisation of objects, that outputs an iterable, that gives correspondance,
#     between bbox (bounding boxes) , generated by the selective search algorithm,
#     and the labelled bounding boxes from the csv
# """
#
# from torch.utils.data import Dataset
# import pandas as pd,numpy as np
#
# from PIL import Image,ImageDraw
# from pathlib import Path
# dir_file = Path(__file__).parent
#
# try:
#     from Code.utils import bbox_mod
#     from Data import data_manager
# except:
#     from Code.utils import bbox_mod
#     from Data import data_manager
#
#
# class DatasetRCNN(Dataset):
#     def __init__(self,mode="train"):
#         """initisation of the dataset
#             :mode : can be equal to train, val or test
#         """
#
#         self.path_csv = self.read_detections_file(mode)
#         self.data_accessor = data_manager.DataAccessor(mode)
#         self.gdtruth_as_list = utils.read_rpos_csv(self.path_csv)
#
#
#
#     def read_detections_file(self,mode):
#         return data_manager.get_path_detecion_csv_filtered_with_rpropos(mode)
#
#     def get_pil_image_from_name(self,image_id):
#         return self.data_accessor.get_pil_image_from_name(image_id)
#
#     def __len__(self):
#         return len(self.gdtruth_as_list)
#
#
#     def __getitem__(self, idx):
#
#         imageID,bboxesInpts,bboxesOutpts,labelsOupts  = self.gdtruth_as_list[idx]
#
#         pil_img = self.get_pil_image_from_name(imageID)
#
#         col,row = pil_img.size
#
#         bboxes_gd = []
#         labels = []
#         for row_df,el in df.iterrows():
#
#             bbox = ((el.XMin*col,el.YMin*row),(el.XMax*col,el.YMax*row))
#             label_name = data_manager.code_to_name_of_class[el.LabelName]
#
#             bbox = bbox_mod.Bbox.from_extremes(*bbox)
#             bboxes_gd.append(bbox)
#             labels.append(label_name)
#
#
#         # TODO : create a function of function (decorator),that
#         # transforms a function that takes a limited number of arguments
#         # for example pil_img and bbox, to a function, that takes as input,
#         # all the arguments pil_img,bboxes_gd and labels
#         # maybe using a dictionnary instead
#
#         if self.joint_transform:
#             pbboxes,bboxes_gd,ious = self.joint_transform(pil_img,bboxes_gd)
#
#             return imageID,pbboxes,bboxes_gd,labels
#         else:
#             return pil_img,bboxes_gd,labels
#
#     def draw_bbox_on_image(self,idx):
#         img, bboxes,labels = self[idx]
#         for bbox in bboxes:
#             bbox.draw_on_image(img,False)
#         img.show()
#
# #TODO : to be tested
#
# if __name__ == '__main__':
#     from Code import custom_transforms
#     from Code import utils
#
#
#     alg = DatasetRCNN()
#     # bbox_associator_inst = custom_transforms.BboxAssociator()
#
#     # import sys;sys.exit()
#     # mode = "train"
#     # re_path_csv = Path(f"../Data/data/{mode}/labels/detections_filtered_with_region_proposals.csv")
#
#     # b = utils.read_csv_and_eval(path_csv)
#     # a = utils.read_rpos_csv(path_csv)