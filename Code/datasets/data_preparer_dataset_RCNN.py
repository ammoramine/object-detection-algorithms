import pandas as pd
from pathlib import Path

try:
    from ...Data import data_manager
    from .. import utils
except:
    from Data import data_manager
    from Code import utils



class DataPreparer_dataRCNN:
    """
        read the csv file, containins the correspondances between the proposed bboxes
        and the labelled bboxes, (name of object with parameters of)

        and create a new csv file, containings the following labels:
        * ID of image
        * proposed bbox
        * associated labelled bbox
        * label of the proposed bbox (can be equal to "background" if iou below a certain value
        of label of the groundTruth bbox)
        * offset between the associated bbox, and the ground Truth bbox, expressed in terms
        as a ratio to the shape of the whole image :
        TODO : test with relative difference insted , replace the shape of the image, by
        the shape of the target bbox
    """
    def __init__(self,mode,truncate = None):
        self.mode = mode
        self.path_csv = self.get_path_csv()
        self.background_label = "Background"
        self.iou_thresh =  0.3
        self.truncate= truncate



        self.data = self.read_serialized_data()

        # import pdb
        self.data = self.update_labels()
        self.data["offsets"] = self.get_offsets()
        # self.data = pdb.runcall(self.update_labels)


    def get_path_csv(self):
        path_csv = data_manager.get_path_detecion_csv_filtered_with_rpropos(self.mode)
        return path_csv
    def read_serialized_data(self):
        """read and convert the data of path_csv_ to dataframe format"""
        res = utils.read_rpos_csv(self.path_csv,self.truncate)

        res = pd.DataFrame(res,columns=["imageID","p_bboxes","gd_bboxes","labels_gd_bboxes"])
        return res

    def get_updated_labels_for_image(self,bbox_inpt,bbox_oupt,labels_oupt):
        return [label if el1.get_iou(el2) > self.iou_thresh else self.background_label for (el1, el2, label) in zip(bbox_inpt,bbox_oupt,labels_oupt)]

    def get_updated_labels(self):
        df = self.data
        return [self.get_updated_labels_for_image(a,b,c) for (a,b,c) in zip(df.p_bboxes,df.gd_bboxes,df.labels_gd_bboxes)]

    def update_labels(self):
        self.data["labels"] = self.get_updated_labels()

        res = self.data.drop("labels_gd_bboxes", axis=1)
        return res

    def get_offsets_for_idx(self,idx):
        a = self.data.p_bboxes[idx]
        b = self.data.gd_bboxes[idx]
        f = data_manager.DataAccessor(mode=self.mode)
        c = f.get_pil_image_from_name(self.data.imageID[0])
        offsets = [aa.get_rel_diff_img(bb,c) for aa,bb in zip(a,b)]
        return offsets

    def get_offsets(self):
        res = [self.get_offsets_for_idx(idx) for idx,el in self.data.iterrows()]
        return res

    # def save(self):
        # re_path_csv = Path(f"../../Data/data/{mode}/labels/detections_for_RCNN.csv")


if __name__ == '__main__':
    mode = "train"
    truncate = 10
    path_csv = data_manager.get_path_detecion_csv_filtered_with_rpropos(mode)
    res = utils.read_rpos_csv(path_csv)

    alg = DataPreparer_dataRCNN(mode,truncate=truncate)

    df = alg.read_serialized_data()
    # res = [[label if el1.get_iou(el2) > 0.3 else self.background_label for (el1, el2, label) in zip(a, b, c)] for (a,b,c) in zip(df.p_bboxes,df.gd_bboxes,df.labels_gd_bboxes)]

    # self.data["labels"] = self.get_updated_labels()
    #
    # self.data.drop("labels_gd_bboxes",inplace=True,axis=1)
