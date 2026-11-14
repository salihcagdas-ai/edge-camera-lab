"""Notification tests: no SMTP connection, credentials or real images."""

from __future__ import annotations

import logging
import ssl
from email.message import EmailMessage

import pytest

from edge.notify import EmailNotifier, NullNotifier


async def test_null_notifier_does_not_retain_or_log_private_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    notifier = NullNotifier()
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            await notifier.send("PRIVATE_SUBJECT", "PRIVATE_BODY", b"PRIVATE_IMAGE")
    assert notifier.sent == []
    assert "PRIVATE_" not in caplog.text
    collector = NullNotifier(capture_for_tests=True)
    await collector.send("example", "test", b"fake")
    assert collector.sent == [("example", "test", True)]


async def test_smtp_failure_logs_only_error_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    notifier = EmailNotifier("example.invalid", 587, "", "", "", "")

    def fail(*_: object) -> None:
        raise OSError("PRIVATE_RECIPIENT_OR_SERVER_RESPONSE")

    monkeypatch.setattr(notifier, "_send_sync", fail)
    with caplog.at_level(logging.WARNING):
        await notifier.send("PRIVATE_SUBJECT", "PRIVATE_BODY", b"PRIVATE_IMAGE")
    assert "OSError" in caplog.text
    assert "PRIVATE_" not in caplog.text


@pytest.mark.parametrize("port", [465, 587])
def test_email_requires_tls_before_login_without_network(
    port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.notify as notify

    calls = []

    class FakeSMTP:
        def __init__(self, host: str, chosen_port: int, **kwargs: object) -> None:
            assert host == "smtp.example.invalid" and chosen_port == port
            assert kwargs["timeout"] == 30
            if port == 465:
                assert isinstance(kwargs["context"], ssl.SSLContext)
                calls.append("tls")

        def __enter__(self) -> FakeSMTP:
            return self

        def __exit__(self, *_: object) -> None:
            calls.append("close")

        def starttls(self, *, context: ssl.SSLContext) -> None:
            assert isinstance(context, ssl.SSLContext)
            calls.append("tls")

        def login(self, user: str, password: str) -> None:
            assert calls == ["tls"]
            calls.append("login")

        def send_message(self, message: EmailMessage) -> None:
            assert message["Subject"] == "Synthetic event"
            assert len(list(message.iter_attachments())) == 1
            calls.append("send")

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(notify.smtplib, "SMTP_SSL", FakeSMTP)
    notifier = EmailNotifier(
        "smtp.example.invalid",
        port,
        "demo@example.invalid",
        "synthetic-password",
        "demo@example.invalid",
        "test@example.invalid",
    )
    notifier._send_sync("Synthetic event", "Synthetic body", b"fake JPEG bytes")
    assert calls == ["tls", "login", "send", "close"]
