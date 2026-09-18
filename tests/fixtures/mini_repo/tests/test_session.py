from miniapp.auth.session import SessionExpired, create_session, validate_session


def test_roundtrip():
    s = create_session("u1", "k", ttl=60)
    assert validate_session(s.token, "k").user_id == "u1"


def test_expired():
    s = create_session("u1", "k", ttl=-1)
    try:
        validate_session(s.token, "k")
    except SessionExpired:
        return
    raise AssertionError("expected SessionExpired")
