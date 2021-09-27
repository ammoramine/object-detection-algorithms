"""moduel containing one class the DataAccessor class """
import pandas as pd
from PIL import Image

try:
    from . import data_manager
except:
    from Data import data_manager


class DataAccessor:
    """
        gives for each mode =  "train", "test" or "validation"
        the possibility to return the csv file as a pandas dataframe,
        or to acess an image as a PIL image
        """

    def __init__(self,mode):
        self.mode = mode

        self.path_to_labels_csv = data_manager.get_path_detection_csv_filtered(self.mode)
        assert self.path_to_labels_csv.exists(), "should launch function on script label_filtering" \
                                                 "to create new csv file"
        self.path_image_folder = data_manager.get_path_image_folder(self.mode)

    def read_csv_file(self):
        # TODO: add python script to create from datasetRCNN, a new csv file, offline,or to
        # or to naviguate trought a large csv file
        return pd.read_csv(self.path_to_labels_csv)

    def get_pil_image_from_name(self, image_id):
        path_img = self.path_image_folder.joinpath(image_id + ".jpg")
        assert path_img.exists(),f"image_id doesn't exists on the folder {self.path_image_folder}"
        pil_image = Image.open(path_img)
        # print(path_img)
        return pil_image
