"""moduel containing one class the DataAccessor class """
import pandas as pd
from PIL import Image

try:
    from .. import data_manager
    from .. import utils_data_reading
except:
    from Data import data_manager,utils_data_reading


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

    def read_data_prepared_for_RCNN(self):
        """
        read csv file, containing data directly used by the neural network of the RCNN model
        :return: data represented as a dataframe
        """
        path_csv = data_manager.get_path_detections_csv_for_RCNN_dataset(self.mode)
        assert path_csv.exists(),f"should launch data_preparer_dataset_RCNN.py for mode {self.mode}"
        df = utils_data_reading.read_csv_and_eval(path_csv,collumns_eval=["p_bboxes",'gd_bboxes','labels','offsets'])
        return df

    def read_data_of_region_crs(self,truncate=None):
        """
        csv file, containing correspondances, between bbox extracted with a region proposal
        algorithm, and labeelled bboxes
        :param truncate: if int read only the first "truncate" rows,if None, read all the lines
        :return:
        """

        path_csv = data_manager.get_path_detecion_csv_filtered_with_rpropos(self.mode)
        assert path_csv.exists(),f"should launch data_preparer_dataset_RCNN.py for mode {self.mode}"

        res = utils_data_reading.read_rpos_csv(path_csv,truncate)

        res = pd.DataFrame(res,columns=["imageID","p_bboxes","gd_bboxes","labels_gd_bboxes"])
        return res

    def get_pil_image_from_name(self, image_id):
        path_img = self.path_image_folder.joinpath(image_id + ".jpg")
        assert path_img.exists(),f"image_id doesn't exists on the folder {self.path_image_folder}"
        pil_image = Image.open(path_img)
        # print(path_img)
        return pil_image
