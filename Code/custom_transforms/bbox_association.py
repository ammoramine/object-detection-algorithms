from . import  selective_search


class BboxAssociator:
    def __init__(self,iou_thresh= 0.5):
        self.iou_thresh = iou_thresh
        self.alg_select = selective_search.SelectiveSearch()

    def associate(self,bboxes_gd,proposed_bboxes):
        return [(bbox,pbbox) for bbox in bboxes_gd for pbbox in proposed_bboxes if bbox.get_iou(pbbox) > self.iou_thresh]

    def __call__(self, img,bboxes_gd):
        proposed_bboxes = self.alg_select(img)
        associations =  self.associate(bboxes_gd,proposed_bboxes)
        return associations

    #TODO : add tests, visual one ?