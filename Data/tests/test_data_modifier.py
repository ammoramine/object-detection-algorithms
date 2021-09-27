import pytest
import pandas as pd,numpy as np

from ..data_modifier import detections_per_class_filtering
from .. import data_manager

# pytest.mark.thisone
def test_offline_data_filtering():
    #TODO: add multiple combinations of classnames:
    classes_names = ["Bus", "Truck"]
    for mode in ["train","test","validation"]:
        detections_per_class_filtering.rewrite_detections_csv_after_filtering(mode,classes_names = classes_names)

        path_csv = data_manager.get_path_detection_csv_filtered(mode)

        df = pd.read_csv(path_csv)

        codes = list(np.unique(df.LabelName))

        assert set(data_manager.get_names_from_codes(codes)) == set(classes_names)