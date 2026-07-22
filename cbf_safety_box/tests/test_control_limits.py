from cbf_safety_box.constraints.control_limits import bounds_from_config


def test_bounds():
    lo, hi = bounds_from_config([-1,-2], [1,2], 2)
    assert lo[0] == -1
    assert hi[1] == 2
