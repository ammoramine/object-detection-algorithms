# """module containins one class the CSVIterator"""
# import csv,itertools
#
# from collections import namedtuple
# from collections.abc import Iterable
#
#
# class CSVIterable(Iterable):
#     """
#         create an iterable over the lines of a csv file
#
#         :inputs:
#         :csv_path: path to csv file
#         :filter_func: function that returns boolean value a csv line,
#          all the lines, that returns false for function, are skipped from the iterable
#
#         Remarks: this class is usefull for large csv file
#         the first line must be a list of strings composed of names
#
#     """
#     def __init__(self,csv_path,col_name_to_type = None,filter_func=None):
#         self.csv_path = csv_path
#         self.col_name_to_type = col_name_to_type
#         self.filter_func = filter_func
#         self.csv_data,self.line_info = self.read_csv_file()
#         self.Labels = namedtuple("Labels",self.line_info) # name of datatype for storage of csv file
#
#     def read_csv_file(self):
#         """
#             open the csv file for reading, and returns an iterator, and header line, for
#             returning a more elaborate iterator
#             :returns:
#              data_csv: iterator over csv lines, (each line is a list)
#              line_info: name of each element of the list
#         """
#         self.csv_file = open(self.csv_path,"r")
#         csv_data = csv.reader(self.csv_file,delimiter=",")
#         line_info = next(csv_data)
#         return csv_data,line_info
#
#     def __iter__(self):
#         """
#
#         :returns:
#             an iterator that outputs a namedTuple, that contains correspondance
#             between name of collumn of the csv file, and the value of the element
#             .... should have used pandas instead...,but never mind
#
#         :remarks: if the file is closes, you must call back the read_csv_file, of the current class
#
#         """
#         self.reset_iterator()
#         type_per_position = [self.col_name_to_type[el] for el in self.line_info]
#         func_cast = lambda el : [cast(el1)  for el1,cast in zip(el,type_per_position)]
#
#         func = lambda el : self.Labels(*el) # function that convert a lines into a Labels object
#
#         csv_iterator = self.csv_data
#         csv_iterator = map(func_cast,csv_iterator)
#         csv_iterator = map(func,csv_iterator)
#
#         if self.filter_func is not None:
#             csv_iterator = itertools.filterfalse(self.filter_func, csv_iterator)
#         return csv_iterator
#
#     def reset_iterator(self):
#         self.csv_data,self.line_info = self.read_csv_file()
#
#     def produce_filtered_csv(self,path_csv):
#         """
#
#         :return: csv_file containing only the rows that satisfies the condition of filter_func
#
#         """
#         filtered_rows = [el for el in self]
#         csv_writer = csv.writer(open(path_csv, "w", newline="\n"))
#
#         csv_writer.writerow([*alg.line_info])
#         for row in filtered_rows:
#             csv_writer.writerow([*row])
#
#     #TODO: test the get_item method
#     def __getitem__(self, idx):
#         iterator = iter(self)
#
#         for i in range(idx+1):
#             try:
#                 el = next(iterator)
#             except StopIteration:
#                 print("choose a smaller index")
#                 return
#         return el
#
# if __name__ == '__main__':
#     csv_path = "Data/data/train/labels/detections.csv"
#     col_name_to_type = {'ImageID': str,
#                         'Source': str,
#                         'LabelName': str,
#                         'Confidence': int,
#                         'XMin': float,
#                         'XMax': float,
#                         'YMin': float,
#                         'YMax': float,
#                         'IsOccluded': int,
#                         'IsTruncated': int,
#                         'IsGroupOf': int,
#                         'IsDepiction': int,
#                         'IsInside': int,
#                         'XClick1X': float,
#                         'XClick2X': float,
#                         'XClick3X': float,
#                         'XClick4X': float,
#                         'XClick1Y': float,
#                         'XClick2Y': float,
#                         'XClick3Y': float,
#                         'XClick4Y': float}
#     alg = CSVIterable(csv_path,col_name_to_type=col_name_to_type)
#     alg_iterator = iter(alg)
#
