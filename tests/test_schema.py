import pytest

from aiku_hands.eval.rollout_schema import RolloutRecord


def test_valid_record():
    r = RolloutRecord(ckpt_step=70000, model="A", condition="nominal", position_id=4, trial=1,
                      seed=1004, success=True, n_objects_placed=3).to_record()
    assert r["failure_tag"] is None and r["position_id"] == 4


@pytest.mark.parametrize("kw", [
    dict(success=False, failure_tag=None),
    dict(success=False, failure_tag="bad_tag"),
    dict(success=True, failure_tag="grasp_pos"),
    dict(position_id=10, success=True),
    dict(model="Z", success=True),
])
def test_invalid(kw):
    base = dict(ckpt_step=1, model="A", condition="nominal", position_id=1, trial=1, seed=1)
    base.update(kw)
    with pytest.raises(ValueError):
        RolloutRecord(**base).validate()
