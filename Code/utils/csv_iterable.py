"""module containins one class the CSVIterator"""
import csv
from collections import namedtuple
from collections.abc import Iterable

class CSVIterable(Iterable):
    """
        create an iterable over the lines of a csv file

        Remarks: this class is usefull for large csv file
    """
    def __init__(self,csv_path):
        self.csv_path = csv_path

        self.csv_data_accessor,self.line_info = self.read_csv_file()

    def read_csv_file(self):
        """
            :returns:
             data_csv: iterator over csv lines, (each line is a list)
             line_info: name of each element of the list
        """
        csv_file = open(self.csv_path,"r")
        data_csv = csv.reader(csv_file,delimiter=",")
        line_info = next(data_csv)
        return data_csv,line_info

    def get_iterator_over_csv_file(self):
        """

        :returns:
            an iterator that outputs a namedTuple, that contains correspondance
            between name of collumn of the csv file, and the value of the element
            .... should have used pandas instead...,but never mind
        """
        Labels = namedtuple("Labels",self.line_info)
        func = lambda el : Labels(*el)

        return map(func,self.csv_data_accessor),Labels
    def __iter__(self):
        self.csv_iterator,self.Labels = self.get_iterator_over_csv_file()

        return self.csv_iterator

        # next(self.csv_iterator)


    # def __call__(self):
    #     """
    #
    #     :return: the iterator over the csv file
    #     """
    #     return self.csv_iterator

