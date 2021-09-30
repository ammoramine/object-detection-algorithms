"""contains one class PathManager, that gives the full path to specific elements of the data
 folder"""
from pathlib import Path
from enum import Enum
dir_file = Path(__file__).parent

path_to_data_folder_default = dir_file.parent.parent.joinpath("Data")

class Mode(Enum):
    train = 1
    test = 2
    validation = 3

def _assert_type_mode(mode):
    message = f"mode should initiated as one of this way : \n"
    for el in Mode.__members__.keys():
        message += f"Mode.{el} \n"
    assert isinstance(mode,Mode),message

class PathManager:
    """
    class that gives path to specific file on the data Directory
    """
    path_to_data_folder = path_to_data_folder_default
    def __init__(self,mode=Mode.train):
        """
            takes as input, an attribute that specifies,
            if the data is from training testing or validation database
            mode could be element from Mode Enum ; Mode.train,Mode.test,Mode.validation
            of from "train","test","validation"
        """
        if isinstance(mode,str):
            mode = getattr(Mode,mode)
        _assert_type_mode(mode)
        self.mode_str = mode.name


    def get_path_detection_csv(self):
        path_detections_csv = self.path_to_data_folder.joinpath(f"data/{self.mode_str}/labels/detections.csv")
        return path_detections_csv

    def get_path_detection_csv_filtered(self):
        return self.replace_detection_csv_ext(ext_replace="_filtered.csv")

    def get_path_detecion_csv_filtered_with_rpropos(self):
        return self.replace_detection_csv_ext(ext_replace="_filtered_with_region_proposals.csv")

    def get_path_detections_csv_for_RCNN_dataset(self):
        return self.replace_detection_csv_ext(ext_replace="_filtered_ready_for_RCNN.csv")

    def replace_detection_csv_ext(self,ext_replace):
        path = Path(str(self.get_path_detection_csv()).replace(".csv", ext_replace))
        return path

    def get_path_folder_images(self):
        path_folder = self.path_to_data_folder.joinpath(f"data/{self.mode_str}/data")
        return path_folder

    def get_path_label_code_corres_csv(self):
        path_csv = self.path_to_data_folder.joinpath(f"data/{self.mode_str}/metadata/classes.csv")
        return path_csv



if __name__ == '__main__':
    alg = PathManager()
# path_detections_csv = dir_file.parent.parent.joinpath(f"Data/data/{mode}/labels/detections.csv")
