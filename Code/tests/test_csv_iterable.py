# import pytest
# import sys
# # try:
# #     from ..utils import csv_iterable
# # except:
# #     from Code.utils import csv_iterable
#
# from pathlib import Path
#
# dir_file = Path(__file__).parent
# # sys.path.append(str(dir_file.joinpath("../Code")))
#
# from Code.utils import csv_iterable
# # @pytest.fixture
#
#
# def get_path_csv(mode="train"):
#     path_csv = dir_file.joinpath(f"../../Data/data/{mode}/labels/detections.csv")
#     return path_csv
#
# @pytest.fixture
# def csv_path_train():
#     path_csv = get_path_csv("train")
#     return path_csv
#
# @pytest.fixture
# def filter_func():
#     def label_filterer(el):
#         return el.LabelName not in ['/m/0dzct']
#     return label_filterer
#

@pytest.mark.slow
def test_if_filter_func_works(csv_path_train,filter_func):
    """
        test if the remaining elements of the iterable , returns false for the
        filtering function
    """
    csv_class = csv_iterable.CSVIterable(csv_path_train,filter_func=filter_func)
    for el in csv_class:
        assert filter_func(el) is False


def test_if_new_produced_csv_is_the_same(filter_func):
    """in the case where no new filtet_func is created"""
    csv_path =  dir_file.joinpath("data_for_tests/detections_for_test.csv")
    csv_path_duplicate = Path(str(csv_path).replace("data_for_tests","data_for_tests_tmp"))
    import shutil
    shutil.copy(csv_path,csv_path_duplicate)
    a,b = csv_path,csv_path_duplicate
    alg_a = csv_iterable.CSVIterable(a, filter_func)
    alg_b = csv_iterable.CSVIterable(b, filter_func)

    for ela,elb in zip(alg_a,alg_b):
        assert ela == elb


