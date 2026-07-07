from treetune.gear.segment_index import SegmentBST


def test_find_nearest_basic():
    bst = SegmentBST()
    bst.insert(-5.0, "a")
    bst.insert(-1.0, "b")
    bst.insert(2.0, "c")
    assert bst.find_nearest(-1.4)[1] == "b"
    assert bst.find_nearest(-100.0)[1] == "a"
    assert bst.find_nearest(100.0)[1] == "c"


def test_empty():
    bst = SegmentBST()
    assert bst.find_nearest(0.0) is None


def test_ties_pick_one():
    bst = SegmentBST()
    bst.insert(0.0, "a")
    bst.insert(0.0, "b")
    out = bst.find_nearest(0.0)
    assert out[1] in ("a", "b")
