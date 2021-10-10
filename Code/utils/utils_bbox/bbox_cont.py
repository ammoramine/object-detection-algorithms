from Code.utils.utils_bbox import bbox_mod

import numpy as np

class BboxCont:
    """
        class that contains an element of the class Bbox, and also
        a container of the bbox, that is specified with its shape, thus the name
        BboxCont, 'Cont' for container.
        The container refers to an image

    """
    @classmethod
    def generate_random_with(cls,const_shape):
        """generate a bbox inside container (img) of shape const_shape"""
        tmp = np.random.rand(4)
        height,width = const_shape
        x, y, w, h = tmp[0] * width,tmp[1] * height,tmp[2] * width,tmp[3] * height

        res = cls.construct_from_simple_params(x,y,w,h,const_shape)
        return res
    @classmethod
    def construct_from_simple_params(cls,*params):
        """more explicitly from the params (x,y,width,height,const_shape)"""
        x,y,width,height,const_shape = params
        res = cls(bbox_mod.Bbox(x, y, width, height), const_shape)
        res.check_if_inside_cont()
        return res

    def __init__(self,bbox,cont_shape):
        """


        :param bbox: of Bbox type, specifying a bbox
        :param cont_shape: shape of the image

        :remarks the parameter cont_shape gives context to the image
        the associations contains a list of objects of same type, for which the center
        of each bbox of the list, is inside the bbox

        """
        self.bbox = bbox
        self.cont_shape = cont_shape
        self.associations = []
        assert self.check_if_inside_cont()

    def __iter__(self):
        return self.bbox.__iter__()

    def __repr__(self):
        message = f"BboxCont(bbox = {self.bbox},cont_shape = {self.cont_shape})"
        return message
    def get_new_bbox_for_new_shape(self,tgt_shape):
        x, y, width, height = self.bbox
        ratio_width = tgt_shape[1]/self.cont_shape[1]
        ratio_height = tgt_shape[0]/self.cont_shape[0]
        x *= ratio_width
        y *= ratio_height
        width *= ratio_width
        height *= ratio_height
        new_bbox = bbox_mod.Bbox(x, y, width, height)
        return new_bbox

    def set_bbox_for_new_shape(self,tgt_shape):
        self.bbox = self.get_new_bbox_for_new_shape(tgt_shape)
        self.cont_shape = tgt_shape

    # def get_relative_position(self,other):
    #     """
    #         of 'other'
    #         get bounding box other,with TP position, expressed as offset of self
    #         and width,height, expressed as ratio to the height and width of the whole image
    #     """
    #
    #     assert isinstance(other,type(self)),f"{other} must be of same type as self"
    #     self.check_same_cont_shape(other)
    #
    #     x1, y1, width1, height1 = self.bbox
    #     x2, y2, width2, height2 = other.bbox
    #
    #     res = bbox_mod.Bbox((x2 - x1) / width1, (y2 - y1) / height1, width2 / self.cont_shape[1], height2 / self.cont_shape[0])
    #
    #     for el in [res.x,res.y]:
    #         assert (el>=0)*(el<1)
    #     for el in [res.width,res.height]:
    #         assert (el>=0)*(el<=1)
    #
    #     return res
    def check_same_cont_shape(self,other):
        assert self.cont_shape == other.cont_shape,"can't compare different BboxCont, with different" \
                                                 "cont_shape"
    def check_inclusion(self,other):
        """
        check if center of other (other.bbox.x,other.bbox.y), is included in self.bbox
        :param other: of type BboxCont, same as self
        :return: True if the test succeds , False otherwise
        """

        self.check_same_cont_shape(other)

        return self.bbox.check_inclusion(other.bbox)

    def check_if_inside_cont(self):
        tmp1 = (self.bbox.x >= 0)  * (self.bbox.y >= 0)
        tmp2 = (self.bbox.x_max <= self.cont_shape[1])
        tmp3 = (self.bbox.y_max <= self.cont_shape[0])

        return tmp1*tmp2*tmp3

    def test_and_add_bbox_with_img(self,other):
        """
        check if center of other is included in self, then add it to associations list of self
        and returns the results of the test (True if included, False else)
        :param other: of type BboxCont, same as self
        :return: boolean value, the result of the test
        """
        test = self.check_inclusion(other)
        if test:
            self.associations.append(other)
        return test

    def reset_associations(self):
        self.associations = []