from . import  selective_search

import numpy as np

class BboxAssociator:
    """
        class containins __init__ method and __call__ methods, that
        computes for a particular image, a list of proposed regions (bounding boxes),
        and associate to each one , one of the labelled bounding boxes

    """
    def __init__(self):
        self.alg_select = selective_search.SelectiveSearch()
        self.background = "background"
        # self.class_names = [self.background] + list(class_names)

    def associate(self,bboxes_gd,proposed_bboxes):
        """

        :param bboxes_gd: list of the labbeled bbox : bounding boxes
        :param proposed_bboxes: list of the proposed bboxes, computed
        from the selective search algorithms
        :return: 3 lists, with corresponding values:
        pbboxes_crs : proposed bboxes from the selective searhc algo,
        bboxes_gd_crs : corresponding labbbeled bbox, with the highest iou ,
        ious_crs : value of the iou
        """

        def func(pbbox):
            iou_max = -np.inf
            for bbox in bboxes_gd:
                iou = pbbox.get_iou(bbox)
                if iou > iou_max:
                    bbox_best = bbox
                    iou_max = iou
            return (pbbox,bbox_best,iou_max)

        pbboxes_crs,bboxes_gd_crs,ious_crs = [],[],[]
        for pbbox in proposed_bboxes:
            pbbox,bbox_best,iou_max = func(pbbox)
            pbboxes_crs.append(pbbox)
            bboxes_gd_crs.append(bbox_best)
            ious_crs.append(iou_max)
        return pbboxes_crs,bboxes_gd_crs,ious_crs

    def __call__(self, img,bboxes_gd):

        """

        :param img: a PIL image
        :param bboxes_gd: bbox pointing to objects on 'img'
        :return: same output as self.associations
        """
        proposed_bboxes = self.alg_select(img)
        associations =  self.associate(bboxes_gd,proposed_bboxes)
        return associations

    #TODO : add tests, visual one ?