from backend.scoring import RiskEngine, RiskLevel


def _engine(**kw):
    defaults = dict(
        window=10, weighting="linear",
        low_threshold=0.40, medium_threshold=0.60, high_threshold=0.75,
        high_consecutive=3, alert_cooldown_seconds=30.0,
    )
    defaults.update(kw)
    return RiskEngine(**defaults)


def test_all_bonafide_stays_none():
    eng = _engine()
    for _ in range(15):
        st = eng.update(0.05, now=0.0)
    assert st.level is RiskLevel.NONE
    assert st.score < 0.4


def test_low_and_medium_bands():
    eng = _engine()
    st = None
    for _ in range(10):
        st = eng.update(0.5, now=0.0)
    assert st.level is RiskLevel.LOW
    eng = _engine()
    for _ in range(10):
        st = eng.update(0.65, now=0.0)
    assert st.level is RiskLevel.MEDIUM


def test_high_needs_three_consecutive_above_threshold():
    eng = _engine()
    # ramp the rolling avg up first
    for _ in range(10):
        eng.update(0.9, now=0.0)
    st1 = eng.update(0.9, now=0.0)
    assert st1.consecutive_high >= 3
    assert st1.level is RiskLevel.HIGH


def test_two_high_then_drop_does_not_latch():
    eng = _engine(window=3, high_consecutive=3)
    eng.update(0.9, now=0.0)
    eng.update(0.9, now=0.0)
    st = eng.update(0.1, now=0.0)
    assert st.consecutive_high == 0
    assert st.level is not RiskLevel.HIGH


def test_alert_fires_once_on_transition():
    eng = _engine()
    alerts = []
    for i in range(20):
        st = eng.update(0.95, now=0.0)
        alerts.append(st.alert)
    assert alerts.count(True) == 1
    assert alerts.index(True) >= 2  # not on the very first chunk


def test_alert_cooldown_blocks_immediate_retrigger():
    eng = _engine(alert_cooldown_seconds=30.0)
    saw_first = any(eng.update(0.95, now=0.0).alert for _ in range(12))
    assert saw_first
    # drop out of HIGH, then climb back within the cooldown window
    eng.update(0.0, now=1.0)
    saw_second = any(eng.update(0.95, now=5.0).alert for _ in range(12))
    assert not saw_second
