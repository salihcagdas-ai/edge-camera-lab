"""Keep test settings independent of the operator's camera and mail configuration."""

import os
import socket
import threading

import pytest


@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in os.environ:
        if name.startswith("EDGE_CAMERA_"):
            monkeypatch.delenv(name)

    original_pair = socket.socketpair
    original_connect = socket.socket.connect
    local_state = threading.local()

    def socketpair(*args, **kwargs):
        # Windows implements asyncio's internal self-pipe with loopback TCP.
        # Permit only the standard library's socketpair construction on this thread.
        local_state.creating_pair = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            local_state.creating_pair = False

    def connect(sock, address):
        if getattr(local_state, "creating_pair", False):
            return original_connect(sock, address)
        raise AssertionError("Tests must not connect to external services")

    def no_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not connect to external services")

    monkeypatch.setattr(socket, "socketpair", socketpair)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", no_connection)
    monkeypatch.setattr(socket, "create_connection", no_connection)
