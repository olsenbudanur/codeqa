from miniapp.api.app import create_app


def test_health():
    assert create_app().dispatch({"path": "/health"})["status"] == 200
