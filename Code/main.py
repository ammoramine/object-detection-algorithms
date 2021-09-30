# import selectivesearch
# from skimage.segmentation import felzenszwalb
# from torch_snippets import *
# # import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# dir_file = os.path.dirname(__file__)
#
#
# def extract_candidates(img):
#     img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
#     img_area = np.prod(img.shape[:2])
#     candidates = []
#     for r in regions:
#         if r['rect'] in candidates: continue
#         if r['size'] < (0.05*img_area): continue
#         if r['size'] > (1*img_area): continue
#         # x, y, w, h = r['rect']
#         candidates.append(list(r['rect']))
#     return candidates
#
# def get_iou(boxA, boxB, epsilon=1e-5):
#     x1 = max(boxA[0], boxB[0])
#     y1 = max(boxA[1], boxB[1])
#     x2 = min(boxA[2], boxB[2])
#     y2 = min(boxA[3], boxB[3])
#     width = (x2 - x1)
#     height = (y2 - y1)
#     if (width<0) or (height <0):
#         return 0.0
#     area_overlap = width * height
#     area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     area_combined = area_a + area_b - area_overlap
#     iou = area_overlap / (area_combined+epsilon)
#     return iou
#
# if __name__ == '__main__':
#     pass
#     # path_img = os.path.join(dir_file,'../Data/Hemanvi.jpeg')
#     #
#     # pil_img = Image.open(path_img)
#     # img = np.array(pil_img)
#     # segments_fz = felzenszwalb(img, scale=200,min_size=100)
#     #
#     # img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
#     #
#     # # fig,axs = plt.subplots(1,3)
#     #
#     # # axs[0].imshow(img)
#     # # axs[1].imshow(segments_fz)
#     # # axs[2].imshow(img_lbl[...,3])
#     # #
#     # # plt.show()
#     # nb_regions = len(regions)
#     # labels_over_segmentation = len(np.unique(img_lbl[...,3]))
#     #
#     # assert nb_regions > labels_over_segmentation
#     #
#     # candidates = extract_candidates(img)# each candidate ahs the following params x, y, w, h
#     # show(img, bbs=candidates)