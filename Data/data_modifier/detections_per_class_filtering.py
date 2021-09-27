"""
    routines that reads a csv file with specified columns'name
    and filter all the rows that belongs to a specific label
"""

"""
bahavior
gets one function, to read the data_repertory
"""
import pandas as pd
from pathlib import Path

# from Data.data_manager import utils
from .. import  data_manager

def rewrite_detections_csv_after_filtering(mode,classes_names,ext_replace="_filtered.csv"):
    accepted_labels_code = data_manager.get_codes_from_names(classes_names)
    path_to_labels_csv = data_manager.get_path_detection_csv(mode)
    q = pd.read_csv(path_to_labels_csv)
    tmp = q.LabelName.isin(accepted_labels_code)
    q = q.loc[tmp]
    new_path_csv = Path(str(path_to_labels_csv).replace(".csv", ext_replace))
    q.to_csv(new_path_csv)





