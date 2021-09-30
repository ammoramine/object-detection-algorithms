
try:
    from ...Data import data_manager
    from .. import utils
except:
    from Data import data_manager



if __name__ == '__main__':
    mode = "train"
    path_csv = data_manager.get_path_detecion_csv_filtered_with_rpropos(mode)
    res = utils.read_rpos_csv(path_csv)