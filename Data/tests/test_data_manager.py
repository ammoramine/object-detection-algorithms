"""
test if the the Data folder verify the conditions
"""
# try:
from ..data_manager import structure_path_data,code_label_corres,data_accessor
from .. import data_manager
# except:
#     from Data.data_manager import structure_path_data

import PIL,pytest

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
    """
    tests if the dictionnary from code to names , and from names to codes are bijective
    for each mode
    :return: nothing
    """
    for mode in ["train", "test", "validation"]:
        alg = code_label_corres.CodeLabelCorres(mode)
        assert alg.code_to_name_of_class == data_manager.code_to_name_of_class
        assert alg.name_to_code_of_class == data_manager.name_to_code_of_class

        for name,code in alg.name_to_code_of_class.items():
            assert data_manager.code_to_name_of_class[code] == name
            assert data_manager.name_to_code_of_class[name] == code

@pytest.mark.slow
def test_data_accessor():
    """
        test if the image refered on the csv file, are the same image on the folder

    """
    for mode in ["train", "test", "validation"]:
        # test if IDimage on each row of csv exists on folder

        alg  = data_accessor.DataAccessor(mode)
        images_ID = alg.read_csv_file().ImageID.unique()

        for image_ID in images_ID:
            pil_image = alg.get_pil_image_from_name(image_ID)

        # test if there is same number of image on csv files, and on folder

        assert len(images_ID) == len(list(alg.path_image_folder.glob("*.jpg")))

