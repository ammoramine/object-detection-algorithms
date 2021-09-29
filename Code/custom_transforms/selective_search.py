import selectivesearch
import numpy as np
from torchvision import transforms
from functools import partial

try:
    from Code.utils import bbox_mod
except:
    from Code.utils import bbox_mod

class SelectiveSearch:
    def __init__(self,scale=200,min_size=100,min_ratio_region=0.05):
        self.scale = scale
        self.min_size = min_size
        self.min_ratio_region = min_ratio_region
        # self.selective_search = self.get_selective_search()

    def __call__(self,pil_img,with_filtering=True):

        self.img_lbl,self.regions = self.apply_selective_search(pil_img)
        if with_filtering:
            self.regions = self.filter_regions()
        return self.regions

    def apply_selective_search(self,pil_img):
        try:
            img = np.array(pil_img)
            if len(img.shape) == 2:
                img = np.stack([img]*3,axis=-1)
            img_lbl, regions = selectivesearch.selective_search(img, scale=self.scale, min_size=self.min_size)
            regions = [bbox_mod.Bbox(*region['rect']) for region in regions]
            return  img_lbl,regions
        except:
            raise ValueError("pil_img, should be 2D with 1 channel or 3D with 3 channel")
    def filter_regions(self):
        img_area = np.prod(self.img_lbl.shape[:2])
        candidates = []
        #TODO BBOX: function to compute area of  the bbox

        for bbox in self.regions:
            if bbox in candidates : continue
            if bbox.area < self.min_ratio_region*img_area : continue
            if bbox.area > img_area : continue
            candidates.append(bbox)
        return candidates

    #TODO BBOX: function to compute area of  the bbox

    def get_iou(self,bboxA, bboxB, epsilon=1e-5):

        return bboxA.get_iou(bboxB,epsilon)


if __name__ == '__main__':
    # img, lbl1, bbox1 = alg[1]
    # img, lbl2, bbox2 = alg[2]

    alg_select = SelectiveSearch()

    # pass
    # # path_img = os.path.join(dir_file,'../Data/Hemanvi.jpeg')
    # #
    # # pil_img = Image.open(path_img)
    # # img = np.array(pil_img)
    # # segments_fz = felzenszwalb(img, scale=200,min_size=100)
    # #
    # # img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    # #
    # # # fig,axs = plt.subplots(1,3)
    # #
    # # # axs[0].imshow(img)
    # # # axs[1].imshow(segments_fz)
    # # # axs[2].imshow(img_lbl[...,3])
    # # #
    # # # plt.show()
    # # nb_regions = len(regions)
    # # labels_over_segmentation = len(np.unique(img_lbl[...,3]))
    # #
    # # assert nb_regions > labels_over_segmentation
    # #
    # # candidates = extract_candidates(img)# each candidate ahs the following params x, y, w, h
    # # show(img, bbs=candidates)