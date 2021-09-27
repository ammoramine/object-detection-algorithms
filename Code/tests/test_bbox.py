import pytest

from ..utils import bbox_mod

def test_if_intersetion_works():
    #intersection should be null
    bbox1 = bbox_mod.Bbox(x=283, y=96, width=430, height=201)
    bbox2 = bbox_mod.Bbox(x=719, y=139, width=303, height=139)

    assert bbox1.intersection(bbox2) == 0
    assert bbox1.intersection(bbox1) == bbox1.get_area()
    assert bbox2.intersection(bbox2) == bbox2.get_area()

    bbox1 = bbox_mod.Bbox(x=0, y=0, width=100, height=100)
    bbox2 = bbox_mod.Bbox(x=50, y=30, width=100, height=100)

    assert bbox1.intersection(bbox2) == 50*70
    assert bbox1.intersection(bbox1) == bbox1.get_area()
    assert bbox2.intersection(bbox2) == bbox2.get_area()