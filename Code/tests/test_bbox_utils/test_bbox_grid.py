try:
    from Code.utils import bbox_grid,bbox_cont
except:
    from Code.utils import bbox_grid,bbox_cont
import pytest

@pytest.fixture
def bboxes_cont():
    """
    generate a list of random bboxes_cont,
    :return:
    """
    N = 0
    tgt_shape = (100,100)
    bboxes_cont = []
    while N<100:
        try:
            tmp = bbox_cont.BboxCont.generate_random_with(tgt_shape)
        except AssertionError:
            continue
        else:
            bboxes_cont.append(tmp)
            N+=1
    return bboxes_cont


@pytest.fixture
def bbox_grid_inst(bboxes_cont):
    """
        increase the value of B, until it finds a value for which the max number of associations
        per grid cell is below B
    """
    S = 7

    B = 1
    while True:
        try:
            bbox_grid_inst = bbox_grid.BboxGrid( S, B, img_shape=(100,100), nb_classes=3)
            bbox_grid_inst.associate_gd_bboxes(bboxes_cont) #necessary to get the value of B
        except ValueError:
            B += 1
            print(B)
        else:
            break

    return bbox_grid_inst



def test_if_associations_are_done_once_per_grid_cell(bboxes_cont,bbox_grid_inst):

    bbox_grid_inst.associate_gd_bboxes(bboxes_cont)
    assert bbox_grid_inst.gd_bboxes_cont == bboxes_cont
    associations = []
    [associations.extend(el.associations) for el in  bbox_grid_inst.grid]

    assert len(associations) == len(bbox_grid_inst.gd_bboxes_cont)

    assert set(associations) == set(bbox_grid_inst.gd_bboxes_cont)

    bbox_grid_inst.reset_associations()

    assert hasattr(bbox_grid,"gd_bboxes_cont") == False

