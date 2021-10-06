try:
    from ..datasets import dataset_YOLO
except:
    from Code.datasets import dataset_YOLO

import torch


def get_data_set_yolo(**kwargs):
    dataset = dataset_YOLO.DatasetYOLO(**kwargs)
    return dataset

def truncate_dataset(dataset,N):
    dataset = torch.utils.data.random_split(dataset, lengths=[N, len(dataset) - N])[0]
    return dataset