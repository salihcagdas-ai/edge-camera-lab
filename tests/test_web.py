"""Web UI/API tests via the FastAPI TestClient (fakes — no camera/cv2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.config import EdgeSettings
from edge.embedder import FakeEmbedder
from edge.hub import FrameHub
from edge.matcher import MatcherRef, OpenSetMatcher
from edge.presence import PresenceState
from edge.store import HomeStore
from edge.web import AppState, _mjpeg_stream, _valid_token, create_app


def _make_client(tmp_path: Path, web_pin: str = "") -> tuple[TestClient, AppState]:
    settings = EdgeSettings(data_dir=tmp_path, embedder="fake", smtp_host="", web_pin=web_pin)
    store = HomeStore(tmp_path)
    state = AppState(
        settings=settings,
        store=store,
        presence=PresenceState(store),
        matcher_ref=MatcherRef(OpenSetMatcher([], settings.match_max_distance)),
        embedder=FakeEmbedder(),
        hub=FrameHub(),
        decode_image=lambda data: np.full((4, 4, 3), 10, dtype=np.uint8) if data else None,
        encode_jpeg=lambda frame: b"jpg",
        web_pin=web_pin,
        session_secret="test-secret",
    )
    return TestClient(create_app(state)), state


def test_health_is_public_and_does_not_leak_state(tmp_path: Path) -> None:
    client, _state = _make_client(tmp_path, web_pin="1234")
    data = client.get("/health").json()  # reachable without login
    assert data == {"status": "ok"}  # no mode / gallery / auth-posture leak


def test_mode_roundtrip_and_validation(tmp_path: Path) -> None:
    client, _state = _make_client(tmp_path)
    assert client.get("/mode").json() == {"mode": "home"}
    assert client.post("/mode", json={"mode": "away"}).json() == {"mode": "away"}
    assert client.get("/mode").json() == {"mode": "away"}
    assert client.post("/mode", json={"mode": "vacation"}).status_code == 422


def test_enroll_updates_gallery_and_people(tmp_path: Path) -> None:
    client, state = _make_client(tmp_path)
    response = client.post(
        "/enroll",
        data={"name": "Demo Person", "is_owner": "true"},
        files=[
            ("photos", ("a.jpg", b"photo-bytes", "image/jpeg")),
            ("photos", ("b.jpg", b"more-bytes", "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["enrolled"] == 2
    assert body["failed"] == []
    assert state.matcher_ref.current.size == 2

    people = client.get("/people").json()
    assert people == [
        {"id": body["person_id"], "name": "Demo Person", "is_owner": True, "embeddings": 2}
    ]


def test_enroll_requires_a_name(tmp_path: Path) -> None:
    client, _state = _make_client(tmp_path)
    response = client.post(
        "/enroll",
        data={"name": "   "},
        files=[("photos", ("a.jpg", b"x", "image/jpeg"))],
    )
    assert response.status_code == 422


def test_events_and_snapshot_serving(tmp_path: Path) -> None:
    client, state = _make_client(tmp_path)
    relative = state.store.save_snapshot(b"jpeg-bytes")
    state.store.log_event("unknown_person", snapshot=relative)
    events = client.get("/events").json()
    assert events[0]["kind"] == "unknown_person"
    assert events[0]["snapshot"] == f"/snapshots/{relative}"
    served = client.get(events[0]["snapshot"])
    assert served.status_code == 200
    assert served.content == b"jpeg-bytes"


def test_index_page_serves_ui(tmp_path: Path) -> None:
    client, _state = _make_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Edge Camera Lab" in response.text


def test_delete_person_rebuilds_matcher(tmp_path: Path) -> None:
    client, state = _make_client(tmp_path)
    body = client.post(
        "/enroll",
        data={"name": "Ali"},
        files=[("photos", ("a.jpg", b"photo-bytes", "image/jpeg"))],
    ).json()
    assert state.matcher_ref.current.size == 1
    response = client.delete(f"/people/{body['person_id']}")
    assert response.status_code == 200
    assert client.get("/people").json() == []
    assert state.matcher_ref.current.size == 0  # gallery rebuilt without them
    assert client.delete("/people/9999").status_code == 404


def test_events_include_clip_and_clip_serving(tmp_path: Path) -> None:
    client, state = _make_client(tmp_path)
    clip_dir = tmp_path / "clips" / "20260702"
    clip_dir.mkdir(parents=True)
    (clip_dir / "olay.mp4").write_bytes(b"mp4-bytes")
    event_id = state.store.log_event("unknown_person")
    state.store.set_event_clip(event_id, "clips/20260702/olay.mp4")

    events = client.get("/events").json()
    assert events[0]["clip"] == "/clips/clips/20260702/olay.mp4"
    served = client.get(events[0]["clip"])
    assert served.status_code == 200
    assert served.content == b"mp4-bytes"
    # Traversal outside data_dir must 404.
    assert client.get("/clips/../secret.mp4").status_code == 404


def test_media_endpoints_cannot_serve_the_database(tmp_path: Path) -> None:
    # The DB lives at data_dir/camera.db (NOT under snapshots/ or clips/), so
    # the /snapshots and /clips endpoints must refuse to serve it even though
    # it is a real file under data_dir. (Regression: the biometric-DB leak.)
    client, _state = _make_client(tmp_path)
    assert (tmp_path / "camera.db").is_file()
    assert client.get("/clips/camera.db").status_code == 404
    assert client.get("/snapshots/camera.db").status_code == 404
    assert client.get("/clips/../camera.db").status_code == 404


def test_pin_gates_every_data_route(tmp_path: Path) -> None:
    client, state = _make_client(tmp_path, web_pin="4271")
    state.store.log_event("unknown_person")
    # /stream returns an infinite MJPEG body, so it is never fully read here.
    # Unauthenticated, the guard raises 401 BEFORE streaming starts, so a plain
    # GET completes normally — that is the case we assert. The authenticated
    # 200 path shares the exact same guard dependency as the routes below.
    protected = ["/events", "/people", "/mode"]
    for path in [*protected, "/stream"]:
        assert client.get(path).status_code == 401, path
    # Index serves the login page (not the app) when unauthenticated.
    assert "Giriş" in client.get("/").text and "canlı" not in client.get("/").text.lower()
    # Wrong PIN rejected, no cookie set.
    assert client.post("/login", json={"pin": "0000"}).status_code == 401
    assert client.get("/events").status_code == 401
    # Correct PIN sets the session cookie; every (non-streaming) route now works.
    assert client.post("/login", json={"pin": "4271"}).json() == {"ok": True}
    for path in protected:
        assert client.get(path).status_code == 200, path
    assert "canlı yayın" in client.get("/").text  # app UI now served
    # Logout clears it.
    client.post("/logout")
    assert client.get("/events").status_code == 401


def test_forged_session_cookie_is_rejected(tmp_path: Path) -> None:
    client, _state = _make_client(tmp_path, web_pin="4271")
    client.cookies.set("edge_camera_session", "9999999999.deadbeef")  # bad shape/signature
    assert client.get("/events").status_code == 401


def test_login_is_rate_limited(tmp_path: Path) -> None:
    client, _state = _make_client(tmp_path, web_pin="4271")
    for _ in range(5):
        assert client.post("/login", json={"pin": "0000"}).status_code == 401
    # 6th attempt is locked out — even the CORRECT pin is refused during lockout.
    assert client.post("/login", json={"pin": "0000"}).status_code == 429
    assert client.post("/login", json={"pin": "4271"}).status_code == 429


def test_logout_revokes_outstanding_tokens_server_side(tmp_path: Path) -> None:
    client, _state = _make_client(tmp_path, web_pin="4271")
    client.post("/login", json={"pin": "4271"})
    stolen = client.cookies.get("edge_camera_session")  # e.g. a sniffed/copied token
    assert client.get("/events").status_code == 200
    client.post("/logout")
    # Re-presenting the pre-logout token must fail: the server bumped its epoch.
    client.cookies.set("edge_camera_session", stolen)
    assert client.get("/events").status_code == 401


def test_unauthenticated_logout_cannot_revoke_another_session(tmp_path: Path) -> None:
    client, state = _make_client(tmp_path, web_pin="4271")
    client.post("/login", json={"pin": "4271"})
    cookie = client.cookies.get("edge_camera_session")
    client.cookies.clear()
    assert client.post("/logout").status_code == 401
    assert state.session_epoch == 0
    client.cookies.set("edge_camera_session", cookie)
    assert client.get("/events").status_code == 200


def test_invalid_nonascii_signature_fails_closed() -> None:
    assert not _valid_token("1.0." + "é" * 64, "test-secret", 0, 1)
    assert not _valid_token("x" * 300, "test-secret", 0, 1)


async def test_active_stream_stops_when_authorization_is_revoked() -> None:
    hub = FrameHub()
    hub.publish(np.zeros((4, 4, 3), dtype=np.uint8))
    allowed = True
    stream = _mjpeg_stream(hub, lambda frame: b"jpeg", lambda: allowed)
    assert b"jpeg" in await anext(stream)
    allowed = False
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.parametrize("limit", [-1, 0, 501])
def test_event_limit_is_bounded(tmp_path: Path, limit: int) -> None:
    client, _state = _make_client(tmp_path)
    assert client.get(f"/events?limit={limit}").status_code == 422


def test_enrollment_size_limit_does_not_leave_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("edge.web._MAX_UPLOAD_FILE_BYTES", 3)
    client, state = _make_client(tmp_path)
    response = client.post(
        "/enroll",
        data={"name": "Visitor"},
        files=[("photos", ("large.jpg", b"1234", "image/jpeg"))],
    )
    assert response.status_code == 413
    assert state.store.people() == []


def test_chunked_enrollment_body_is_limited_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("edge.web._MAX_UPLOAD_REQUEST_BYTES", 32)
    client, state = _make_client(tmp_path)
    response = client.post(
        "/enroll",
        headers={"content-type": "multipart/form-data; boundary=example"},
        content=iter([b"--example\r\n", b"x" * 64]),
    )
    assert response.status_code == 413
    assert state.store.people() == []


def test_enrollment_rejects_ambiguous_group_photo(tmp_path: Path) -> None:
    class GroupEmbedder(FakeEmbedder):
        def embed_all(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
            face = super().embed_all(frame)
            return face + face

    client, state = _make_client(tmp_path)
    state.embedder = GroupEmbedder()
    response = client.post(
        "/enroll",
        data={"name": "Visitor", "is_owner": "true"},
        files=[("photos", ("group.jpg", b"group-photo", "image/jpeg"))],
    )
    assert response.status_code == 200
    assert response.json()["enrolled"] == 0
    assert response.json()["person_id"] is None
    assert state.store.people() == []


def test_failed_enrollment_does_not_create_a_person(tmp_path: Path) -> None:
    client, state = _make_client(tmp_path)
    state.decode_image = lambda data: None
    response = client.post(
        "/enroll",
        data={"name": "Visitor"},
        files=[("photos", ("invalid.jpg", b"not-a-photo", "image/jpeg"))],
    )
    assert response.json()["enrolled"] == 0
    assert state.store.people() == []
