# """
#     dataset object (data streamer) over image with corresponding boundinx box, with
#     labelisation of objects
# """
#
# from torch.utils.data import Dataset
# from pathlib import Path
# import pandas as pd
#
# try:
#     from Data import code_label_corres
# except:
#     from Data.data_manager import code_label_corres
# dir_file = Path(__file__).parent
#
# class CSVFilterer(Dataset):
#     """
#         class that read a csv and reatain only specific elements
#     """
#     def __init__(self,mode="train",classes_names = ("Bus","Truck")):
#         """initisation of the dataset
#             :inputs:
#             :mode : can be equal to train, val or test
#         """
#     ######## init options
#         self.mode = mode
#         self.classes_names = classes_names
#         self.repertory_data = self.get_repertory_data_for_mode()
#     ### construct_path_from repertory data
#         self.path_to_labels_csv = self.get_labels_path()
#
#
#     ### construct dictionnaries between names and code of classes of interest (should be done outside)
#     #     self.code_to_name_of_class, self.name_to_code_of_class = self.get_class_name_to_code_convertor()
#     # ### get the code of the accepted labels
#     #     self.accepted_labels_code = [self.name_to_code_of_class[el] for el in self.classes_names]
#         self.accepted_labels_code = code_label_corres.get_from_classe_name()
#
#         self.filtered_data_frame = self.get_filtered_dataframe()
#
#         self.write_csv_after_filtering(new_basename="detections_filtered.csv")
#
#
#
#     def get_repertory_data_for_mode(self):
#         repertory = dir_file.joinpath(f"../Data/data/{self.mode}")
#         assert repertory.exists()
#         return repertory
#     def get_labels_path(self):
#         path = self.repertory_data.joinpath("labels/detections.csv")
#         return path
#
#     def get_filtered_dataframe(self):
#         q = pd.read_csv(self.path_to_labels_csv)
#         tmp = q.LabelName.isin(self.accepted_labels_code)
#         q = q.loc[tmp]
#         return q
#     def write_csv_after_filtering(self,new_basename):
#         new_path_csv = Path(str(self.path_to_labels_csv).replace("detections.csv", new_basename))
#         self.filtered_data_frame.to_csv(new_path_csv)
#
#
#
#     def get_class_name_to_code_convertor(self):
#         f = self.repertory_data.joinpath("metadata/classes.csv")
#         g = pd.read_csv(f, names=["class_code", "class_name"])
#         code_to_name_of_class = dict(zip(g.class_code, g.class_name))
#         name_to_code_of_class = dict(zip(g.class_name, g.class_code))
#         return code_to_name_of_class, name_to_code_of_class
#
#
#
#
# if __name__ == '__main__':
#     alg = CSVFilterer()
#     # self = alg
#     pass
#     # r = self.csv_iterator
#
#
