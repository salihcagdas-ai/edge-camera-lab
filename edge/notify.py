"""Notifications. Email (SMTP) is the agreed channel — alarms and recognitions
go to the owner's phone-connected inbox, snapshot attached.

``EmailNotifier`` uses stdlib smtplib; blocking I/O is pushed off the event
loop with ``asyncio.to_thread``. ``NullNotifier`` (tests / SMTP unconfigured)
degrades to logging so the system stays usable without mail credentials.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Protocol

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    async def send(self, subject: str, body: str, jpeg: bytes | None = None) -> None: ...


class NullNotifier:
    """Drop notification contents unless an explicit test collector is requested."""

    def __init__(self, *, capture_for_tests: bool = False) -> None:
        self._capture_for_tests = capture_for_tests
        self.sent: list[tuple[str, str, bool]] = []  # inspected by tests

    async def send(self, subject: str, body: str, jpeg: bytes | None = None) -> None:
        if self._capture_for_tests:
            self.sent.append((subject, body, jpeg is not None))
        logger.info("notification skipped: SMTP is not configured")


class EmailNotifier:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        mail_from: str,
        mail_to: str,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._from = mail_from or user
        self._to = mail_to

    def _send_sync(self, subject: str, body: str, jpeg: bytes | None) -> None:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._from
        message["To"] = self._to
        message.set_content(body)
        if jpeg is not None:
            message.add_attachment(jpeg, maintype="image", subtype="jpeg", filename="olay.jpg")
        context = ssl.create_default_context()
        if self._port == 465:  # implicit TLS
            with smtplib.SMTP_SSL(self._host, self._port, context=context, timeout=30) as smtp:
                if self._user:
                    smtp.login(self._user, self._password)
                smtp.send_message(message)
        else:  # STARTTLS (587 default)
            with smtplib.SMTP(self._host, self._port, timeout=30) as smtp:
                smtp.starttls(context=context)
                if self._user:
                    smtp.login(self._user, self._password)
                smtp.send_message(message)

    async def send(self, subject: str, body: str, jpeg: bytes | None = None) -> None:
        try:
            await asyncio.to_thread(self._send_sync, subject, body, jpeg)
        except Exception as error:  # neither recipient nor message content belongs in logs
            logger.warning("email notification failed (%s)", type(error).__name__)
