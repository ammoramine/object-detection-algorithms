# import pytest
#
#
# from .. import dataset_RCNN
#
#
# def check_if_code_to_name_conversion_is_consistent():
#     alg = dataset_RCNN.DatasetRCNN(mode="train")
#     for (key,value) in alg.code_to_name_of_class.items():
#         assert alg.name_to_code_of_class[value] == key
#
#
# # @pytest.mark.slow
# def test_if_filtering_worked():
#     class_names = ("Bus", "Truck")
#     alg = dataset_RCNN.DatasetRCNN(mode="train", classes_names=class_names)
#     # n = 1000
#     # for el,i in zip(alg,range(n)):
#     for el in alg:
#         assert alg.code_to_name_of_class[el.LabelName] in class_names