"""
    the functions present in this script gives main utilities to
    hides the complexities of the other modules of the package
"""
from . import code_label_corres
from . import structure_path_data

Mode = structure_path_data.Mode

def get_path_detection_csv(mode):
    path_manager = structure_path_data.PathManager(mode)
    path_csv = path_manager.get_path_detection_csv()
    return path_csv

def get_path_detection_csv_filtered(mode):
    path_manager = structure_path_data.PathManager(mode)
    path_csv = path_manager.get_path_detection_csv_filtered()
    return path_csv

def get_path_image_folder(mode):
    path_manager = structure_path_data.PathManager(mode)
    path_folder_images = path_manager.get_path_folder_images()
    return path_folder_images

tmp = code_label_corres.CodeLabelCorres()
code_to_name_of_class = tmp.code_to_name_of_class
name_to_code_of_class = tmp.name_to_code_of_class

def get_codes_from_names(codes):
    return [name_to_code_of_class[code] for code in codes]

def get_names_from_codes(codes):
    return [code_to_name_of_class[code] for code in codes]