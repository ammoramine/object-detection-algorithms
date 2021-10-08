try:
    from Code.utils import bbox_grid,bbox_cont
except:
    from Code.utils import bbox_grid,bbox_cont
import torch,numpy as np
from torch.nn.functional import one_hot
class TensorToLbdBBoxesCont:
    """
        a class that can transform a bboxGrid with corepondance
        of bbox to label, into a torch tensor
        (quantification of the data of the bboxGrid) and gives
        also the inverse transform

        The two objects are a way of representing the output of a Yolo Model, or alike models
    """
    def __init__(self,shape_tensor,int_to_label):#bbox_gd_to_labels,labels_to_int
        """

        :param bbox_grid:
        :param bbox_gd_to_labels:
        :param labels_to_int:
        :param nb_classes:
        """
        self.shape_tensor = shape_tensor
        self.int_to_label = int_to_label

        self.bbox_grid_inst = self.construct_grid()
        # self.labels_to_int = labels_to_int
        # self.nb_classes = len(labels_to_int)
        #
        # self.shape_tensor = self.bbox_grid.S, self.bbox_grid.S, 5 * self.bbox_grid.B + self.nb_classes

        self.squeezed_shape_tensor = (np.product(self.shape_tensor[:2]),self.shape_tensor[-1])

    def construct_grid(self):
        pass

    def __call__(self,target_tensor):
        """

        :param target_tensor: there is two case it is of same shape as sekf.shape_tensor or of
        same shape as (N,*self.shape_tensor)
        :return:
        """
        assert target_tensor.shape == self.shape_tensor
        self.shape_tensor = self.shape_tensor.reshape(-1,self.shape_tensor[-1])
        grid = []
        gd_to_label = dict()
        for el in self.shape_tensor:
            xs,yx,widths,heights,objectnesses = [el[i:5*self.bbox_grid.B:5] for i in range(5)]
            idx = torch.argmax(el[5*self.bbox_grid.B:],axis=-1)
            label = self.int_to_label[idx]
            for (x,y,w,h,objectness) in zip(xs,yx,widths,heights,objectnesses):
                if objectness>0.5:
                    bbox_cont_inst = bbox_cont.BboxCont.construct_from_simple_params(x,y,w,h,self.bbox_grid.img_shape)
                    grid.append(bbox_cont_inst)
                    gd_to_label[bbox_cont_inst] = label
        gd_bboxes_cont = grid
        return gd_bboxes_cont,gd_to_label

