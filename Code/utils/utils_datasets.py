try:
    from ..datasets import dataset_YOLO
except:
    from Code.datasets import dataset_YOLO

DatasetYOLO = dataset_YOLO.DatasetYOLO

def get_data_set_yolo(**kwargs):
    dataset = dataset_YOLO.DatasetYOLO(**kwargs)
    return dataset
