try:
    from Code.utils import bbox_cont
    from Code.utils.utils_bbox import bbox_mod
except:
    from Code.utils import bbox_mod,bbox_cont


class BboxGrid:
    """
        a class that partionnate and image to a grid of SxS bbox, covering the image
        and that provides routines to associate  for an iterable of bboxes , the
        unique element of the grid, that contains the center of the bbox.
    """
    def __init__(self,S,B,img_shape = (448,448)):
        self.S = S
        self.B = B
        self.img_shape = img_shape
        # self.nb_classes = nb_classes

        self.grid = self.construct_grid()

    def __iter__(self):
        return iter(self.grid)

    def construct_grid(self):
        """of disjoint bboxes, equally spaced """
        stride_row = self.img_shape[0]/self.S
        stride_col = self.img_shape[1]/self.S

        grid = []
        for i in range(self.S):
            for j in range(self.S):
                # print(i,j)
                bbox = bbox_mod.Bbox(stride_col * j, stride_row * i, stride_col, stride_row)
                bbox_cont_inst = bbox_cont.BboxCont(bbox, self.img_shape)
                grid.append(bbox_cont_inst)
        return grid

    def associate_gd_bboxes(self,gd_bboxes_cont):
        """
        associates to each labelled bbox_cont :  gd_bboxes_cont, a grid cell
        and makes sure  that it is associated only once

        :param gd_bboxes_cont: list of BboxCont objects
        :return: None
        :remarks should be done once, thus the private method

        """
        self.reset_associations()
        self.gd_bboxes_cont = gd_bboxes_cont

        for gd_bbox in self.gd_bboxes_cont:
            for grid_cell in self.grid:
                test = grid_cell.test_and_add_bbox_with_img(gd_bbox)
                if test:
                    # it test succeed go out of the loop, because gd_bbox can only be associated
                    # once to the grid cell
                    break
        self.check_max_associations_per_cell()

    def check_max_associations_per_cell(self):
        """
        of the grid doesn't exceed the value self.B, and raises ValueError if it's the case
        :return:
        """
        for grid_cell in self.grid:
            if len(grid_cell.associations) > self.B:
                # each grid_cell, have a limited number of associations
                raise ValueError(f"can't associate more than self.B == {self.B} elements"
                                 f"to the grid_cell, chek the dataset,"
                                 f"then change the value of self.B , to higher value of "
                                 f"necessary")

    def reset_associations(self):
        if hasattr(self,"gd_bboxes_cont"):
            delattr(self,"gd_bboxes_cont")
        for grid_cell in self.grid:
            grid_cell.reset_associations()



if __name__ == '__main__':
    pass
    # alg = BboxGrid(S=7,B=2)
    # alg.construct_grid()