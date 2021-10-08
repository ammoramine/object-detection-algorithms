import pandas as pd
from .utils_bbox import bbox_mod

BboxParams = bbox_mod.BboxParams
Bbox = bbox_mod.Bbox


def transforms_iterable_to_idxs(iterable):
    """
    and returns
    :param iterable: of objects of arbitrary type
    :return: iterable_idxs : iterable transforms to indexes
            idx_to_iterable : a dictionnary that transforms an index to an iterable
    """
    iterable_pruned = remove_duplicate_of_iterable(iterable)
    iterable_to_idx = dict(zip(iterable_pruned,range(len(iterable_pruned))))
    idx_to_iterable = {value:key for (key,value) in iterable_to_idx.items()}
    iterable_idxs = [iterable_to_idx[el]  for el in iterable]
    return iterable_idxs,idx_to_iterable


def remove_duplicate_of_iterable(iterable):
    res = []
    [res.append(x) for x in iterable if x not in res]
    return res


def convert_bboxes_to_bboxes_params(iterable):
    res = []
    for el in iterable:
        imageID, pbboxes, bboxes_gd, labels = el
        bboxes_gd = [el.bbox_params for el in bboxes_gd]
        pbboxes = [el.bbox_params for el in pbboxes]
        res.append((imageID, pbboxes, bboxes_gd, labels))
    return res

def convert_bboxes_params_to_bboxes(iterable):
    res = []
    for el in iterable:
        imageID, pbboxes, bboxes_gd, labels = el
        bboxes_gd = [Bbox(*el) for el in bboxes_gd]
        pbboxes = [Bbox(*el) for el in pbboxes]
        res.append((imageID, pbboxes, bboxes_gd, labels))
    return res

def transform_bboxes_gd_to_idxs(iterable):
    res = []
    for el in iterable:
        imageID, pbboxes, bboxes_gd, labels = el
        bboxes_gd_idxs,idx_to_bbox_gd = transforms_iterable_to_idxs(bboxes_gd)
        res.append((imageID,pbboxes,bboxes_gd_idxs,labels,idx_to_bbox_gd))
    return res


def recover_bboxes_gd_from_idx(iterable):
    res = []
    for el in iterable:
        imageID, pbboxes, bboxes_gd_idxs, labels, idx_to_bbox_gd = el

        bboxes_gd = [idx_to_bbox_gd[idx] for idx in bboxes_gd_idxs]
        labels_per_bbox = [labels[idx] for idx in bboxes_gd_idxs]
        res.append((imageID, pbboxes, bboxes_gd, labels_per_bbox))
    return res

def read_csv_and_eval(path_csv = "producedData/tmp_data/tmp.csv",collumns_eval=("pbboxes","idx_to_bbox","bboxes_gd_idx","labels"),truncate=None):

    if isinstance(truncate,int):
        df = pd.read_csv(path_csv, index_col=0,nrows = truncate)
    else:
        df = pd.read_csv(path_csv, index_col=0)



    for name_col,el in df.items():
        if name_col in  collumns_eval:
            df[name_col] = eval_col_of_df(el)
    return df

def eval_col_of_df(col):
    return [eval(el) for el in col]


def read_rpos_csv(path_csv,truncate=None):

    aa = read_csv_and_eval(path_csv,truncate=truncate)
    res1 = aa.values.tolist()
    res11 = recover_bboxes_gd_from_idx(res1)

    res = convert_bboxes_params_to_bboxes(res11)

    return res