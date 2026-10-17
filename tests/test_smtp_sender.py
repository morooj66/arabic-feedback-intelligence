"""SmtpEmailSender against a LOCAL fake SMTP server (loopback only, nothing leaves the machine), plus the complete password-reset flow delivered through it."""
from __future__ import annotations

import base64
import email
import email.policy
import re
import socketserver
import threading

import pytest
from sqlalchemy.orm import Session

from config import settings
from rcs.review import auth
from rcs.review.auth import REVIEWER, authenticate, confirm_password_reset, create_user, request_password_reset, resolve_session, start_session
from rcs.review.email_sender import OutgoingEmail, SmtpEmailSender, UnconfiguredEmailSender, get_email_sender, sender_from_settings, set_email_sender
from rcs.review.models import PasswordResetToken
from rcs.review.persistence import bundle_from_engine
from rcs.review.repository import get_engine, init_db


class FakeSmtp(socketserver.ThreadingTCPServer):
    """Minimal ESMTP server. `offer_starttls=False` means it never advertises STARTTLS (and, as in reality, a client that insists on TLS must refuse to continue)."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, offer_auth=True):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.offer_auth = offer_auth
        self.messages: list[dict] = []
        self.commands: list[str] = []
        self.logins: list[str] = []


class _Handler(socketserver.StreamRequestHandler):
    def send(self, line):
        self.wfile.write((line + "\r\n").encode()); self.wfile.flush()

    def handle(self):
        s = self.server
        self.send("220 fake ESMTP")
        mail_from, rcpts = None, []
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            cmd = line.split(" ", 1)[0].upper()
            s.commands.append(cmd)
            if cmd == "EHLO":
                self.wfile.write(b"250-fake\r\n"); self.wfile.write(b"250-AUTH PLAIN\r\n" if s.offer_auth else b""); self.wfile.write(b"250 SIZE 100000\r\n"); self.wfile.flush()
            elif cmd == "AUTH":
                s.logins.append(base64.b64decode(line.split(" ")[-1]).decode()); self.send("235 ok")
            elif cmd == "MAIL":
                mail_from = line; self.send("250 ok")
            elif cmd == "RCPT":
                rcpts.append(line); self.send("250 ok")
            elif cmd == "DATA":
                self.send("354 go")
                data = []
                while True:
                    l = self.rfile.readline().decode("utf-8", "replace")
                    if l in (".\r\n", ".\n", ""):
                        break
                    data.append(l[1:] if l.startswith("..") else l)
                s.messages.append({"from": mail_from, "rcpts": rcpts, "raw": "".join(data)}); self.send("250 queued")
            elif cmd == "QUIT":
                self.send("221 bye"); return
            else:
                self.send("250 ok")


@pytest.fixture
def smtp():
    srv = FakeSmtp()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown(); srv.server_close()


def plain_sender(srv, **kw):
    return SmtpEmailSender(host="127.0.0.1", port=srv.server_address[1], security="starttls", username=kw.pop("username", "mailer"), password="pw-1",
                           from_address="no-reply@example.invalid", allow_plaintext=True, **kw)


def parse(msg):
    return email.message_from_string(msg["raw"], policy=email.policy.default)


def test_a_message_is_delivered_with_arabic_subject_body_and_correct_envelope(smtp):
    plain_sender(smtp).send(OutgoingEmail(to="user@example.invalid", subject="إعادة تعيين كلمة المرور", body="افتح الرابط: https://app.example.invalid/reset-password#token=abc\nشكرًا"))
    assert len(smtp.messages) == 1
    m = smtp.messages[0]
    assert "no-reply@example.invalid" in m["from"] and any("user@example.invalid" in r for r in m["rcpts"])
    parsed = parse(m)
    assert parsed["Subject"] == "إعادة تعيين كلمة المرور" and parsed["To"] == "user@example.invalid" and parsed["From"] == "no-reply@example.invalid"
    assert "token=abc" in parsed.get_content() and "شكرًا" in parsed.get_content()
    assert smtp.logins == ["\x00mailer\x00pw-1"]                                            # credentials sent via AUTH PLAIN (over TLS in real use)


def test_no_login_is_attempted_without_a_username(smtp):
    plain_sender(smtp, username="").send(OutgoingEmail("u@example.invalid", "s", "b"))
    assert smtp.logins == [] and len(smtp.messages) == 1


def test_starttls_mode_refuses_a_server_without_tls_and_never_sends_the_message_or_credentials(smtp):
    s = SmtpEmailSender(host="127.0.0.1", port=smtp.server_address[1], security="starttls", username="mailer", password="pw-1", from_address="no-reply@example.invalid")
    with pytest.raises(Exception):
        s.send(OutgoingEmail("u@example.invalid", "s", "SECRET-LINK-BODY"))
    assert smtp.messages == [] and smtp.logins == [] and "MAIL" not in smtp.commands and "DATA" not in smtp.commands    # no plain-text fallback, nothing leaked


def test_implicit_tls_mode_fails_against_a_plain_server_and_leaks_nothing(smtp):
    s = SmtpEmailSender(host="127.0.0.1", port=smtp.server_address[1], security="ssl", username="mailer", password="pw-1", from_address="no-reply@example.invalid", timeout=3)
    with pytest.raises(Exception):
        s.send(OutgoingEmail("u@example.invalid", "s", "SECRET-LINK-BODY"))
    assert smtp.messages == [] and smtp.logins == []


def test_constructor_validation_and_no_plaintext_switch_in_the_environment(monkeypatch):
    with pytest.raises(ValueError):
        SmtpEmailSender(host="h", port=25, security="none", username="", password="", from_address="a@b")
    with pytest.raises(ValueError):
        SmtpEmailSender(host="", port=25, security="starttls", username="", password="", from_address="a@b")
    monkeypatch.setattr(settings, "SMTP_HOST", "smtp.example.invalid"); monkeypatch.setattr(settings, "SMTP_FROM", "no-reply@example.invalid")
    monkeypatch.setattr(settings, "SMTP_SECURITY", "plain")                                 # an invalid/insecure value is refused: the sender stays unconfigured
    assert isinstance(sender_from_settings(), UnconfiguredEmailSender)
    for attr in ("allow_plaintext",):
        assert not hasattr(settings, "SMTP_ALLOW_PLAINTEXT") and "allow_plaintext" not in open(settings.__file__, encoding="utf-8").read()


def test_the_sender_comes_from_the_environment_settings_only_when_fully_configured(monkeypatch):
    set_email_sender(None)
    monkeypatch.setattr(settings, "SMTP_HOST", ""); monkeypatch.setattr(settings, "SMTP_FROM", "")
    assert isinstance(get_email_sender(), UnconfiguredEmailSender)
    monkeypatch.setattr(settings, "SMTP_HOST", "smtp.example.invalid")                      # host without a from-address is not enough
    assert isinstance(get_email_sender(), UnconfiguredEmailSender)
    monkeypatch.setattr(settings, "SMTP_FROM", "no-reply@example.invalid"); monkeypatch.setattr(settings, "SMTP_SECURITY", "ssl"); monkeypatch.setattr(settings, "SMTP_PORT", 465)
    s = get_email_sender()
    assert isinstance(s, SmtpEmailSender) and (s.host, s.port, s.security, s.allow_plaintext) == ("smtp.example.invalid", 465, "ssl", False) and s.configured


def test_complete_password_reset_flow_delivered_through_smtp(tmp_path, smtp, monkeypatch):
    """request -> real SMTP delivery of the link -> confirm with the token FROM THE EMAIL -> old password dead, new works, old sessions revoked, link single-use."""
    monkeypatch.setattr(auth, "_run_async", lambda fn: fn())
    monkeypatch.setattr(settings, "PASSWORD_RESET_LINK_BASE", "https://rcs.example.invalid")
    auth._reset_attempts.clear()
    engine = get_engine(tmp_path / "r.sqlite"); init_db(engine); bundle = bundle_from_engine(engine)
    user = create_user(bundle, email="reviewer@example.invalid", password="old-password-123", display_name="R", role=REVIEWER)
    cookie = start_session(bundle, user_id=user.user_id)
    set_email_sender(plain_sender(smtp))
    try:
        request_password_reset(bundle, "Reviewer@Example.Invalid")
        assert len(smtp.messages) == 1
        body = parse(smtp.messages[0]).get_content()
        token = re.search(r"/reset-password#token=([A-Za-z0-9_\-]+)", body).group(1)
        assert any("reviewer@example.invalid" in r for r in smtp.messages[0]["rcpts"])
        with Session(engine) as s:
            stored = [r.token_hash for r in s.query(PasswordResetToken).all()]
        assert stored == [auth.hash_reset_token(token)] and token not in stored             # only the HASH of the emailed token is stored
        assert confirm_password_reset(bundle, token, "brand-new-password-9") is True
        assert authenticate(bundle, email="reviewer@example.invalid", password="old-password-123") is None
        assert authenticate(bundle, email="reviewer@example.invalid", password="brand-new-password-9") is not None
        assert resolve_session(bundle, cookie) is None                                       # every earlier session was revoked
        assert confirm_password_reset(bundle, token, "another-password-77") is False         # single use
        request_password_reset(bundle, "unknown@example.invalid")                           # unknown address: nothing sent, nothing differs
        assert len(smtp.messages) == 1
    finally:
        set_email_sender(None)
        auth._reset_attempts.clear()


def test_a_failing_mail_server_does_not_break_the_request_or_leak_details(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(auth, "_run_async", lambda fn: fn())
    monkeypatch.setattr(settings, "PASSWORD_RESET_LINK_BASE", "https://rcs.example.invalid")
    auth._reset_attempts.clear()
    engine = get_engine(tmp_path / "f.sqlite"); init_db(engine); bundle = bundle_from_engine(engine)
    create_user(bundle, email="reviewer@example.invalid", password="old-password-123", display_name="R", role=REVIEWER)
    set_email_sender(SmtpEmailSender(host="127.0.0.1", port=9, security="starttls", username="", password="", from_address="no-reply@example.invalid", timeout=2, allow_plaintext=True))
    try:
        with caplog.at_level("WARNING"):
            assert request_password_reset(bundle, "reviewer@example.invalid") is None
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "reviewer@example.invalid" not in logged and "token=" not in logged and "error_type=" in logged
    finally:
        set_email_sender(None)
        auth._reset_attempts.clear()
