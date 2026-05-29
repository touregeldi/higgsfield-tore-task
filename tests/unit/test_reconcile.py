from memory_service.services.reconcile import plan_reconciliation, Action
from memory_service.models.domain import MemoryCandidate, MemoryType


def _cand(key, value, conf=0.8):
    return MemoryCandidate(type=MemoryType.fact, key=key, value=value, confidence=conf)


def test_new_key_is_insert():
    actions = plan_reconciliation([_cand("location", "Berlin")], existing_active={})
    assert actions[0].kind == Action.INSERT
    assert actions[0].supersedes is None


def test_same_key_different_value_supersedes():
    actions = plan_reconciliation([_cand("location", "Munich")],
                                  existing_active={"location": ("m-old", "Berlin")})
    assert actions[0].kind == Action.INSERT
    assert actions[0].supersedes == "m-old"


def test_same_key_same_value_is_skip():
    actions = plan_reconciliation([_cand("location", "Berlin")],
                                  existing_active={"location": ("m-old", "Berlin")})
    assert actions[0].kind == Action.SKIP
