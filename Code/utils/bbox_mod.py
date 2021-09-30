"""
    *module that contains two class, 'BBoxParams', and 'Bbox' , the first is a named tuple
    containing parameters of a bounding box, the second, contains routines to create new bounding box

    * positions are given by by row followed by col,



the Bbox class should be able to construct new Bbox (bounding box), centered at other location, and with other
scale, and different height and width

it should also be able to

the BBox class, is immutable, but have routines, to create new Bbox, it's only purpose, is to help, the
user to crop images, and to create other usefull instances.
It also check


"""

import numpy as np,cv2

class OnceSettingDescriptor:
    def __init__(self, storage_name):
        print(f"setting descriptor for {storage_name}")
        self.storage_name = storage_name
    def __set__(self, instance, value):
        if instance.initiated_class:
            raise ValueError("can't set any value more than once")
        else:
            instance.__dict__[self.storage_name] = value
            instance.initiated_class = True
    def __get__(self, instance, owner):
        # return getattr(instance, self.storage_name)
        return instance.__dict__[self.storage_name]

from  collections import namedtuple
from PIL import ImageDraw

import os
dirFile = os.path.dirname(__file__)



BboxParams = namedtuple("BboxParams",["x","y","width","height"],defaults=[0]*4)
BboxParams.__doc__= """
    * BBoxParams class is a named Tuple, used  to store parameters of a bounding box ,containing the four usual parameters:
    position of the top left pixel, and the height and width, that squares a region of the image.
    It also contains the 'scale' parameter that is specify if a rescaling must be done, before
    squaring the image.
    If scale is different than one, the bounding box squares a region, on the image which  is resized by
    this factor
    
    Remarks : 
     we  define the absolute height and width, as the height and width at scale 1,
     absolute dimensions for bbox with scale s, are obtained by  the formula (s*height,s*width)
     
     According to the definition of scale, for a bouding box with parameters (x,y,width,height,1),
     the bouding box (x*scale,y*scale,width*scale,height*scale,scale) squares the same regions
     
"""

class Bbox:
    """
        class that contains only one paramter that can't be modified : an instance of 'BBoxParams'

    """
    bbox_params = OnceSettingDescriptor("bbox_params")
    initiated_class = False
    @classmethod
    def from_bbbox_params(cls,bbox_params):
        assert isinstance(bbox_params,BboxParams)
        return cls(*bbox_params)
    @classmethod
    def from_extremes(cls,LT_pt,RB_pt):
        x_min,y_min = LT_pt
        x_max,y_max = RB_pt
        x,y,width,height = x_min,y_min,x_max-x_min,y_max-y_min
        return cls(x,y,width,height)

    def __init__(self,x=0,y=0,width=0,height=0):
        bbox_params = BboxParams(x,y,width,height)
        self.bbox_params = bbox_params

    def __repr__(self):
        return self.bbox_params.__repr__().replace("BboxParams","Bbox")
    def __iter__(self):
        return self.bbox_params.__iter__()
    def __hash__(self):
        return hash(self.x)^hash(self.y)^hash(self.width)^hash(self.height)
    # def __next__(self):
    #     pass
    def __eq__(self, other):
        res = all([self.x == other.x,
                  self.y == other.y,
                   self.width == other.width,
                   self.height == other.height
                  ])
        return res

    def __getattr__(self, item):
        if item in  ['x','y','width','height']:
            res = self.bbox_params.__getattribute__(item)
            return res
        elif item == 'x_max':
            return self.x+self.width
        elif item == 'y_max':
            return self.y+self.height
        elif item == 'x_min':
            return self.x
        elif item == 'y_min':
            return self.y
        elif item == 'area':
            return self.get_area()

        #TODO : understand why the pandas Dataframe object, call the __next__ method
        # of bbox_mod

        # else:
        #     raise ValueError(f"item {item} is not recognized")

    def intersection(self,other_bbox):
        assert isinstance(other_bbox,type(self))
        x1 = max(self.x_min, other_bbox.x_min)
        y1 = max(self.y_min, other_bbox.y_min)

        x2 = min(self.x_max, other_bbox.x_max)
        y2 = min(self.y_max, other_bbox.y_max)
        width = (x2 - x1)
        height = (y2 - y1)
        if (width<0) or (height <0):
            return 0.0
        area_overlap = width * height
        return area_overlap

    def get_iou(self, other_bbox, epsilon=1e-5):

        area_overlap = self.intersection(other_bbox)
        area_a = self.area
        area_b = other_bbox.area
        area_combined = area_a + area_b - area_overlap
        iou = area_overlap / (area_combined+epsilon)
        return iou

    def __sub__(self,other):
        res = np.array([getattr(self,el)-getattr(other,el) for el in ["x","y","x_max","y_max"]])
        return res

    def get_rel_diff_img(self,other,pil_img):
        """
            get the difference, between the TL and BR points with 'other bbox
        """
        diff = self - other
        w,h = pil_img.size
        res = diff / np.array([w,h,w,h])
        res = list(res)
        return res

    def get_area(self):
        area = self.width*self.height
        return area




    def draw_on_image(self,img,with_show=True):
        s = ImageDraw.Draw(img)
        bbox_as_extrems = ((self.x_min,self.y_min),(self.x_max,self.y_max))
        s.rectangle(bbox_as_extrems,width=10,outline="red")
        if with_show:
            img.show()

    def crop_image(self,pil_image):
        """crop the image such that """

        limits = (self.x_min,self.y_min,self.x_max,self.y_max)
        return pil_image.crop(limits)





    # def get_center_bbox(self):
    #     """return center of the bounding box"""
    #     x,y,width,height,_ = self.bbox_params
    #     res = (height/2, width/2)
    #     return res

    # #TODO (ideally): add optional paramter for the
    # def shift_center_bbox(self,relative_position_row_col):
    #     """
    #     shift the center to "relative_position_row_col"
    #     :param: relative_position_row_col containing the row index and col index of
    #     the new position relative to the top-left point of the bbox
    #     """
    #     (y_cen,x_cen) =  self.get_center_bbox()
    #     shift = relative_position_row_col[0]-y_cen,relative_position_row_col[1]-x_cen
    #     return self.construct_new_bbox(y = self.bbox_params.y+shift[0],x = self.bbox_params.x+shift[1])

    # def rescale_bbox(self,scale):
    #     """
    #      change the scale value
    #     """
    #     assert  scale >0
    #     tmp = self.bbox_params._asdict().copy()
    #     old_scale = tmp["scale"]
    #     factor_scale = scale/old_scale
    #     tmp["x"] *= factor_scale
    #     tmp["y"] *= factor_scale
    #     tmp["width"] *= factor_scale
    #     tmp["height"] *= factor_scale
    #     tmp["scale"] = scale
    #     return self.construct_new_bbox(**tmp)
    #
    # def get_back_to_absolute_scale(self):
    #     return self.rescale_bbox(scale=1)
    #
    # def modify_width_and_height(self,new_width,new_height):
    #     """
    #         modify width and height of bbox keeping the same center
    #     """
    #     x,y,width,height,_ = self.bbox_params
    #     new_x = x  + (width - new_width)/2
    #     new_y = y  + (height - new_height)/2
    #     return self.construct_new_bbox(x=new_x, y=new_y, width=new_width, height=new_height)
    #
    # def rescale_bbox_to_new_scale_adding_extending_borders(self,scale,percent_width=0,percent_height=0):
    #     """
    #         change the scale keeping the same center, with new width and height equal to:
    #         width * (1+percent_width)
    #         height * (1+percent_height)
    #     """
    #     #TODO: add the same routine, that extends the absolute height and width instead
    #     new_bbox_instance = self.rescale_bbox(scale)
    #     width_target = self.bbox_params.width * (1+percent_width)
    #     height_target = self.bbox_params.height * (1+percent_height)
    #     new_bbox_instance = new_bbox_instance.modify_width_and_height(new_width=width_target,new_height=height_target)
    #     return new_bbox_instance
    #
    # def construct_new_bbox(self,**kwparams):
    #     s = self.bbox_params._asdict().copy()
    #     s.update(**kwparams)
    #     cls = type(self)
    #     return cls(**s)
    #
    # def get_limits(self):
    #     """
    #         returns a four parameters, the first row of the bounding box, the last row
    #         thr first collum, and the last one
    #     """
    #     a = int(self.y)
    #     b = int(self.y+self.height)
    #
    #     c = int(self.x)
    #     d = int(self.x+self.width)
    #
    #     return a,b,c,d
    #
    #
    # def crop_image(self,image):
    #     """crop the image such that """
    #     if self.scale != 1:
    #         bbox_at_aboslute_scale = self.get_back_to_absolute_scale()
    #     else:
    #         bbox_at_aboslute_scale = self
    #     a,b,c,d = bbox_at_aboslute_scale.get_limits()
    #     cropped_image = image[a:b,c:d]
    #     resized_cropped_image = self.resize_image(cropped_image)
    #     return resized_cropped_image
    #
    # def resize_image(self,image):
    #     return cv2.resize(image,fx=self.scale,fy=self.scale,dsize=(0,0))
    #
    # def draw_or_show_crop_on_image(self,image,draw=True,color=(255,0,0),failure=False):
    #     image = image.copy()
    #     new_bbox = self.get_back_to_absolute_scale()
    #     a, b, c, d = new_bbox.get_limits()
    #     p1 = (c,a)
    #     p2 = (d,b)
    #     if failure:
    #         self.putText(image)
    #     else:
    #         cv2.rectangle(image, p1, p2, color, 2, 1)
    #     if draw:
    #         return image
    #     else:
    #         utils.show_multiple(image)
    #
    # def putText(self,image,sentence = "Failure"):
    #     """ at top left of bounding_box"""
    #     position = (int(self.x),int(self.y))
    #     cv2.putText(image,sentence,position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 255),3)

    # TODO : change scale to tuple is needed