import pytest
from .... import utils

@pytest.fixture
def dataset_ins():
    """of the YOLO"""
    dataset_ins = utils.get_data_set_yolo()
    return dataset_ins



def test_if_bbox_normalization_works(dataset_ins):

