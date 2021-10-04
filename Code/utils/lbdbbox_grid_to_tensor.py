try:
    from . import bbox_grid,bbox_cont
except:
    from Code.utils import bbox_grid,bbox_cont
import torch,numpy as np
from torch.nn.functional import one_hot
class LbdBBoxesContToTensor:
    """
        a class that can transform a bboxGrid with corepondance
        of bbox to label, into a torch tensor
        (quantification of the data of the bboxGrid) and gives
        also the inverse transform

        The two objects are a way of representing the output of a Yolo Model, or alike models
    """
    def __init__(self,bbox_grid,labels_to_int):#bbox_gd_to_labels,labels_to_int
        """

        :param bbox_grid:
        :param bbox_gd_to_labels:
        :param labels_to_int:
        :param nb_classes:
        """
        self.bbox_grid = bbox_grid
        self.labels_to_int = labels_to_int
        self.nb_classes = len(labels_to_int)

        self.shape_tensor = self.bbox_grid.S, self.bbox_grid.S, 5 * self.bbox_grid.B + self.nb_classes

        self.squeezed_shape_tensor = (np.product(self.shape_tensor[:2]),self.shape_tensor[-1])

    def associate(self,gd_bboxes_cont_to_labels):
        gd_bboxes_cont = gd_bboxes_cont_to_labels.keys()
        self.bbox_grid.associate_gd_bboxes(gd_bboxes_cont)
        self.gd_bboxes_cont_to_labels = gd_bboxes_cont_to_labels

    def __call__(self,gd_bboxes_cont_to_labels):
        """
            should be called after the association are done with the method associate_gd_bboxes
            it computes an array the represents the associations of the labbeled bbox, with the grid.
            The concatenation of this array for multiple image, gives the ground truth , from which the
            neural network is fed

        :return: array of shape (self.S,self.S,(5*self.B+self.nb_classes) of same shape
                as the output of the neural network, for a unique image (if you remove the
                batch's size)
        """
        self.associate(gd_bboxes_cont_to_labels)
        outpts = torch.zeros(self.squeezed_shape_tensor)
        for grid_cell,outpt in zip(self.bbox_grid.grid,outpts):
            if len(grid_cell.associations) > 0:
                for i,association in enumerate(grid_cell.associations):
                    x, y, w, h = grid_cell.get_relative_position(association)
                    outpt[5*i+0] = x
                    outpt[5*i+1] = y
                    outpt[5*i+2] = w
                    outpt[5*i+3] = h
                    outpt[5*i+4] = 1
                label = self.gd_bboxes_cont_to_labels[association]
                #TODO: all the associations must have the same label
                # check that during testing

                label_as_int = self.labels_to_int[label]
                outpt[5*self.bbox_grid.B+label_as_int] = 1
        outpts = outpts.reshape(self.shape_tensor)
        return outpts

    # def get_bbox_grid_from_target(self,target):
    #     assert target.shape == self.shape_tensor
    #     self.shape_tensor = self.shape_tensor.reshape(-1,self.shape_tensor[-1])
    #     grid = []
    #     for el in self.shape_tensor:
    #         xs,yx,widths,heights,objectnesses = [el[i:5*self.bbox_grid.B:5] for i in range(5)]
    #         idx = torch.argmax(el[5*self.bbox_grid.B:],axis=-1)
    #         label = self.int_to_labels[idx]
    #         for (x,y,w,h,objectness) in zip(xs,yx,widths,heights,objectnesses):
    #             if objectness>0.5:
    #                 bbox_cont_inst = bbox_cont.BboxCont.construct_from_simple_params(x,y,w,h,self.bbox_grid.img_shape)
    #                 grid.append(bbox_cont_inst)
    #     return grid,label
    #
