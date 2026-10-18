import json

import pytest

from livetrans.core.events import (
    PROTOCOL_VERSION,
    EventFactory,
    error_payload,
    session_state_payload,
)


def test_envelope_fields_and_seq_monotonic():
    f = EventFactory("sess-1")
    e1 = f.make("session.state", session_state_payload("starting"))
    e2 = f.make("error", error_payload("E", "msg", True))
    assert (e1.seq, e2.seq) == (1, 2)
    d = json.loads(e1.to_json())
    assert set(d) == {"v", "type", "seq", "session_id", "ts_wall_ms", "payload"}
    assert d["v"] == PROTOCOL_VERSION
    assert d["session_id"] == "sess-1"
    assert d["payload"]["state"] == "starting"


def test_to_json_keeps_unicode():
    f = EventFactory("s")
    e = f.make("error", error_payload("E", "한국어 메시지", False))
    assert "한국어 메시지" in e.to_json()


def test_unknown_session_state_rejected():
    with pytest.raises(ValueError):
        session_state_payload("flying")
