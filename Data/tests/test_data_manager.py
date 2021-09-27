"""
test if the the Data folder verify the conditions
"""
# try:
from ..data_manager import structure_path_data,code_label_corres
from .. import data_manager
# except:
#     from Data.data_manager import structure_path_data



def test_if_paths_exists():
    for mode in ["train","test","validation"]:
        path_manager = structure_path_data.PathManager(mode)
        path_detection_csv = path_manager.get_path_detection_csv()
        path_folder_images = path_manager.get_path_folder_images()
        path_label_code_corres_csv = path_manager.get_path_label_code_corres_csv()
        assert path_detection_csv.is_file()
        assert path_folder_images.is_dir()
        assert path_label_code_corres_csv.is_file()

        assert path_detection_csv == data_manager.get_path_detection_csv(mode)
        assert path_folder_images == data_manager.get_path_image_folder(mode)

def test_code_label_correspondance():
    for mode in ["train", "test", "validation"]:
        alg = code_label_corres.CodeLabelCorres(mode)
        assert alg.code_to_name_of_class == data_manager.code_to_name_of_class
        assert alg.name_to_code_of_class == data_manager.name_to_code_of_class

        for name,code in alg.name_to_code_of_class.items():
            assert data_manager.code_to_name_of_class[code] == name
            assert data_manager.name_to_code_of_class[name] == code
