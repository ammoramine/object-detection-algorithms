import pandas as pd
from pathlib import Path


dir_file = Path(__file__).parent

from . import structure_path_data

PathManager = structure_path_data.PathManager
Mode = structure_path_data.Mode
# path_detections_csv = dir_file.parent.parent.joinpath(f"Data/data/{mode}/labels/detections.csv")

class CodeLabelCorres:
    def __init__(self,mode=Mode.train):
        self.path_csv = PathManager(mode=mode).get_path_label_code_corres_csv()
        self.code_to_name_of_class, self.name_to_code_of_class = self.construct_dict_correspondance()

    def construct_dict_correspondance(self):
        g = pd.read_csv(self.path_csv, names=["class_code", "class_name"])
        code_to_name_of_class = dict(zip(g.class_code, g.class_name))
        name_to_code_of_class = dict(zip(g.class_name, g.class_code))
        return code_to_name_of_class, name_to_code_of_class


    def get_code_from_labels(self,labels_names):
        codes = [self.code_to_name_of_class[label] for label in labels_names]
        return codes
