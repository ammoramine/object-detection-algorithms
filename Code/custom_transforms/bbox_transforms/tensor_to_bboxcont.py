try:
    from ...utils.utils_bbox import bbox_grid,bbox_cont
except:
    from Code.utils.utils_bbox import bbox_grid,bbox_cont
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
    def __init__(self,int_to_label,bbox_grid):#bbox_gd_to_labels,labels_to_int
        """

        :param bbox_grid:
        :param bbox_gd_to_labels:
        :param labels_to_int:
        :param nb_classes:
        """
        self.int_to_label = int_to_label
        self.bbox_grid = bbox_grid
        self.nb_classes = len(self.int_to_label)


        self.img_shape = self.bbox_grid.img_shape

        self.shape_tensor = (self.bbox_grid.S,self.bbox_grid.S,5*self.bbox_grid.B+self.nb_classes)

        self.squeezed_shape_tensor = (np.product(self.shape_tensor[:2]),self.shape_tensor[-1])


    def parse_tensor(self,target_tensor):
        """

        :param target_tensor:
        :return:  xs,ys,widths,heights,objectnesses : 5 tensor of shape
        """
        xs,ys,widths,heights,objectnesses = [target_tensor[...,5*j:5*(j+1)] for j in range(self.bbox_grid.B)]


    def __call__(self,target_tensor):
        """
        each el of the target tensor encode the relative parameters of a bbox,
        we convert it to the abs parameters
        :param target_tensor: tensor of shape (S,S)
        :return:
        """

        x_offset = np.arange(0,self.bbox_grid.S)*self.bbox_grid.stride_col
        y_offset = np.arange(0,self.bbox_grid.S)*self.bbox_grid.stride_row

        x_offset = np.repeat(x_offset.reshape(1,-1),repeats=self.bbox_grid.S,axis=0)
        y_offset = np.repeat(y_offset.reshape(-1,1),repeats=self.bbox_grid.S,axis=1)
        widths_offset = self.bbox_grid.img_shape[1]
        heights_offset = self.bbox_grid.img_shape[0]

        stride_row = self.bbox_grid.stride_row
        stride_col = self.bbox_grid.stride_col


        # for el in target_tensor:
        # xs,ys,widths,heights,objectnesses = [target_tensor[...,i:5*self.bbox_grid.B:5] for i in range(5)]

        xs *= stride_col
        ys *= stride_row

        xs += x_offset
        ys += y_offset

        widths *= widths_offset
        heights *= heights_offset

        return target_tensor


    def __call__(self,target_tensor):
        """

        :param target_tensor: there is two case it is of same shape as self.shape_tensor or of
        same shape as (N,*self.shape_tensor)

        :return:
        """
        assert target_tensor.shape == self.shape_tensor
        target_tensor = target_tensor.reshape(-1,self.shape_tensor[-1])
        grid = []
        gd_to_label = dict()
        for el in target_tensor:
            xs,yx,widths,heights,objectnesses = [el[i:5*self.bbox_grid.B:5] for i in range(5)]
            idx = (el[5*self.bbox_grid.B:]).argmax(axis=-1)
            label = self.int_to_label[idx]
            for (x,y,w,h,objectness) in zip(xs,yx,widths,heights,objectnesses):
                if objectness>0.5:
                    bbox_cont_inst = bbox_cont.BboxCont.construct_from_simple_params(x,y,w,h,self.bbox_grid.img_shape)
                    grid.append(bbox_cont_inst)
                    gd_to_label[bbox_cont_inst] = label
        gd_bboxes_cont = grid
        return gd_bboxes_cont,gd_to_label

