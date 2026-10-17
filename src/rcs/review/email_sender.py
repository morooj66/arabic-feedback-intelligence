"""Email-sender abstraction for self-service password reset.

Deliberately tiny: one Protocol and two implementations. NO real provider is wired in yet - the default is `UnconfiguredEmailSender`, which makes password reset
FAIL CLOSED (the API answers "unavailable" and no token is ever created). A real sender (SMTP or an API provider) is a later, separate step: implement
`EmailSender.send`, then call `set_email_sender(...)` at startup. `FakeEmailSender` records messages in memory for tests and local runs; it never touches the network.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol


class EmailNotConfigured(RuntimeError):
    """Raised by the unconfigured sender; never carries an address or a message body."""


@dataclass(frozen=True)
class OutgoingEmail:
    to: str
    subject: str
    body: str


class EmailSender(Protocol):
    configured: bool

    def send(self, message: OutgoingEmail) -> None:
        """Deliver one message or raise. Must not log the body (it contains a one-time link)."""
        ...


class UnconfiguredEmailSender:
    configured = False

    def send(self, message: OutgoingEmail) -> None:
        raise EmailNotConfigured("no email provider is configured")


class FakeEmailSender:
    """Test/local double: keeps the outbox in memory, sends nothing."""

    configured = True

    def __init__(self) -> None:
        self.outbox: list[OutgoingEmail] = []
        self._lock = threading.Lock()

    def send(self, message: OutgoingEmail) -> None:
        with self._lock:
            self.outbox.append(message)


class SmtpEmailSender:
    """Real delivery over SMTP with MANDATORY TLS and certificate verification. `security` is "starttls" (the connection is upgraded before any credential or message is
    sent; a server that does not offer STARTTLS is refused) or "ssl" (implicit TLS). There is no plain-text mode reachable from configuration; `allow_plaintext` exists
    only as a constructor argument so tests can talk to a local fake server, and nothing in the environment can set it. The message body (it holds a one-time link) and
    the recipient are never logged here."""

    configured = True

    def __init__(self, *, host: str, port: int, security: str, username: str, password: str, from_address: str, timeout: float = 15.0,
                 allow_plaintext: bool = False, ssl_context=None) -> None:
        if security not in ("starttls", "ssl"):
            raise ValueError("security must be 'starttls' or 'ssl'")
        if not host or not from_address:
            raise ValueError("host and from_address are required")
        self.host, self.port, self.security, self.username, self.password = host, port, security, username, password
        self.from_address, self.timeout, self.allow_plaintext, self.ssl_context = from_address, timeout, allow_plaintext, ssl_context

    def send(self, message: OutgoingEmail) -> None:
        import smtplib
        import ssl
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.from_address, message.to, message.subject
        msg.set_content(message.body, charset="utf-8", cte="base64")      # explicit: Arabic must reach any server intact, 8bit support not assumed
        ctx = self.ssl_context or ssl.create_default_context()
        if self.allow_plaintext:
            smtp = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
        elif self.security == "ssl":
            smtp = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=ctx)
        else:
            smtp = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
        with smtp:
            smtp.ehlo()
            if not self.allow_plaintext and self.security == "starttls":
                smtp.starttls(context=ctx)                                          # raises if the server does not support it: never falls back to plain text
                smtp.ehlo()
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(msg)


def sender_from_settings() -> EmailSender:
    """SmtpEmailSender when RCS_SMTP_HOST and RCS_SMTP_FROM are set and valid, otherwise the fail-closed UnconfiguredEmailSender."""
    from config import settings

    if not (settings.SMTP_HOST and settings.SMTP_FROM):
        return UnconfiguredEmailSender()
    try:
        return SmtpEmailSender(host=settings.SMTP_HOST, port=settings.SMTP_PORT, security=settings.SMTP_SECURITY, username=settings.SMTP_USERNAME,
                               password=settings.SMTP_PASSWORD, from_address=settings.SMTP_FROM)
    except ValueError:
        return UnconfiguredEmailSender()


_sender: EmailSender | None = None          # None = "not installed explicitly": resolved from the environment on use


def get_email_sender() -> EmailSender:
    return _sender if _sender is not None else sender_from_settings()


def set_email_sender(sender: EmailSender | None) -> None:
    """Install the process-wide sender (None restores the default: SMTP from the environment if configured, otherwise fail closed). Called once at startup by whoever wires a real provider, and by tests."""
    global _sender
    _sender = sender
