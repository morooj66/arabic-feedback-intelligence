"""Login rate limiting behind a reverse proxy: trusted-proxy address resolution, spoofed headers, per-account isolation, brute-force caps."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from config import settings
from rcs.review import auth
from rcs.review.auth import (
    LOGIN_LIMIT_PER_ACCOUNT_ADDRESS,
    LOGIN_LIMIT_PER_ADDRESS,
    LOGIN_LIMIT_PER_EMAIL,
    create_user,
    login_attempt_allowed,
    record_failed_login,
    resolve_client_ip,
)
from rcs.review.http_api import serve_in_thread
from rcs.review.repository import get_engine, init_db


@pytest.fixture(autouse=True)
def _clean_limiter():
    auth._rate_limit_attempts.clear()
    yield
    auth._rate_limit_attempts.clear()


# ---- address resolution ------------------------------------------------------------------------------------------------------------
def test_direct_connection_uses_the_socket_peer_and_ignores_the_header():
    assert resolve_client_ip("198.51.100.4", "6.6.6.6", ()) == "198.51.100.4"                 # no trusted proxies configured
    assert resolve_client_ip("198.51.100.4", "6.6.6.6", ("10.0.0.0/8",)) == "198.51.100.4"    # peer is not a trusted proxy
    assert resolve_client_ip("198.51.100.4", None, ()) == "198.51.100.4"


def test_trusted_proxy_forwards_the_real_client():
    assert resolve_client_ip("10.1.2.3", "203.0.113.9", ("10.0.0.0/8",)) == "203.0.113.9"
    assert resolve_client_ip("10.1.2.3", None, ("10.0.0.0/8",)) == "10.1.2.3"                  # no header: the proxy itself
    assert resolve_client_ip("10.1.2.3", "203.0.113.9:5555", ("10.0.0.0/8",)) == "203.0.113.9"  # port suffix tolerated


def test_spoofed_leftmost_entries_are_ignored_even_through_a_trusted_proxy():
    # the client sends "1.1.1.1"; the proxy appends the real peer it saw -> the rightmost untrusted hop is the truth
    assert resolve_client_ip("10.1.2.3", "1.1.1.1, 203.0.113.9", ("10.0.0.0/8",)) == "203.0.113.9"
    assert resolve_client_ip("10.1.2.3", "1.1.1.1, 203.0.113.9, 10.9.9.9", ("10.0.0.0/8",)) == "203.0.113.9"   # chain of two trusted proxies


def test_header_from_an_untrusted_peer_cannot_choose_the_address():
    a = resolve_client_ip("198.51.100.4", "7.7.7.7", ("10.0.0.0/8",))
    b = resolve_client_ip("198.51.100.4", "8.8.8.8", ("10.0.0.0/8",))
    assert a == b == "198.51.100.4"


@pytest.mark.parametrize("header", ["not-an-ip", "203.0.113.9, garbage", ",,,", "999.1.1.1"])
def test_malformed_forwarded_header_falls_back_to_the_proxy(header):
    assert resolve_client_ip("10.1.2.3", header, ("10.0.0.0/8",)) == "10.1.2.3"


def test_malformed_trusted_entries_trust_nothing_and_ipv6_works():
    assert resolve_client_ip("10.1.2.3", "203.0.113.9", ("not-a-cidr",)) == "10.1.2.3"
    assert resolve_client_ip("2001:db8::1", "2001:db8:ffff::5", ("2001:db8::/64",)) == "2001:db8:ffff::5"
    assert resolve_client_ip(None, "1.2.3.4", ("10.0.0.0/8",)) == "unknown"
    assert resolve_client_ip("garbage", "1.2.3.4", ("10.0.0.0/8",)) == "unknown"


def test_all_hops_trusted_returns_the_proxy():
    assert resolve_client_ip("10.1.2.3", "10.5.5.5, 10.6.6.6", ("10.0.0.0/8",)) == "10.1.2.3"


# ---- limiter semantics -------------------------------------------------------------------------------------------------------------
def test_one_users_failures_do_not_lock_out_another_user_behind_the_same_address():
    for _ in range(LOGIN_LIMIT_PER_ACCOUNT_ADDRESS):
        record_failed_login("10.1.2.3", "alice@x.com")
    assert login_attempt_allowed("10.1.2.3", "alice@x.com") is False
    assert login_attempt_allowed("10.1.2.3", "bob@x.com") is True


def test_per_account_limit_still_blocks_brute_force_of_one_account_from_one_address():
    for i in range(LOGIN_LIMIT_PER_ACCOUNT_ADDRESS):
        assert login_attempt_allowed("203.0.113.9", "alice@x.com") is True
        record_failed_login("203.0.113.9", "alice@x.com")
    assert login_attempt_allowed("203.0.113.9", "alice@x.com") is False


def test_email_is_case_and_whitespace_insensitive():
    for _ in range(LOGIN_LIMIT_PER_ACCOUNT_ADDRESS):
        record_failed_login("203.0.113.9", "Alice@X.com ")
    assert login_attempt_allowed("203.0.113.9", "alice@x.com") is False


def test_password_spraying_across_many_emails_from_one_address_is_capped():
    for i in range(LOGIN_LIMIT_PER_ADDRESS):
        record_failed_login("203.0.113.9", f"user{i}@x.com")
    assert login_attempt_allowed("203.0.113.9", "brand-new@x.com") is False
    assert login_attempt_allowed("203.0.113.77", "brand-new@x.com") is True              # a different address is unaffected


def test_distributed_guessing_of_one_account_from_many_addresses_is_capped():
    for i in range(LOGIN_LIMIT_PER_EMAIL):
        record_failed_login(f"203.0.113.{i}", "alice@x.com")
    assert login_attempt_allowed("198.51.100.1", "alice@x.com") is False
    assert login_attempt_allowed("198.51.100.1", "bob@x.com") is True


def test_legacy_key_functions_still_work():
    for _ in range(5):
        auth.record_login_attempt("legacy-key")
    assert auth.check_login_rate_limit("legacy-key") is False


def test_tracked_keys_are_bounded(monkeypatch):
    monkeypatch.setattr(auth, "_MAX_TRACKED_KEYS", 50)
    monkeypatch.setattr(auth, "_RATE_LIMIT_WINDOW_SECONDS", -1)                             # everything counts as expired
    for i in range(200):
        record_failed_login(f"203.0.113.{i % 250}", f"u{i}@x.com")
    assert len(auth._rate_limit_attempts) < 200 * 3


# ---- HTTP level --------------------------------------------------------------------------------------------------------------------
def _post(base, email, password, xff=None):
    headers = {"Content-Type": "application/json"}
    if xff:
        headers["X-Forwarded-For"] = xff
    req = urllib.request.Request(base + "/api/login", data=json.dumps({"email": email, "password": password}).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.headers


@pytest.fixture()
def server(tmp_path):
    engine = get_engine(tmp_path / "rl.sqlite"); init_db(engine)
    create_user(engine, email="alice@x.com", password="alice-password-1", display_name="A", role="REVIEWER")
    create_user(engine, email="bob@x.com", password="bob-password-1", display_name="B", role="ADMIN")
    srv, base = serve_in_thread(engine)
    yield base
    srv.shutdown()


def test_http_one_user_cannot_lock_out_another_behind_a_shared_proxy_address(server):
    for _ in range(8):
        status, _h = _post(server, "alice@x.com", "wrong-password")
    assert status == 429
    status, headers = _post(server, "bob@x.com", "bob-password-1")                            # same socket address, different account
    assert status == 200 and "rcs_session=" in (headers.get("Set-Cookie") or "")


def test_http_brute_force_of_one_account_is_still_blocked_even_with_the_right_password(server):
    for _ in range(5):
        assert _post(server, "alice@x.com", "wrong-password")[0] == 401
    assert _post(server, "alice@x.com", "alice-password-1")[0] == 429


def test_http_spoofed_forwarded_header_does_not_evade_the_limit_when_no_proxy_is_trusted(server, monkeypatch):
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", ())
    statuses = [_post(server, "alice@x.com", "wrong-password", xff=f"9.9.9.{i}")[0] for i in range(8)]
    assert statuses[-1] == 429                                                                # a fresh fake address per attempt buys nothing


def test_http_trusted_proxy_separates_real_clients(server, monkeypatch):
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", ("127.0.0.1",))                         # the test client is the "proxy"
    for _ in range(30):
        _post(server, "bob@x.com", "wrong-password", xff="203.0.113.1")                       # one real client sprays until its address cap
    assert _post(server, "bob@x.com", "bob-password-1", xff="203.0.113.1")[0] == 429
    assert _post(server, "bob@x.com", "bob-password-1", xff="203.0.113.2")[0] == 200           # another real client behind the same proxy is fine


def test_http_login_still_issues_a_working_session_and_roles_are_unchanged(server):
    status, headers = _post(server, "bob@x.com", "bob-password-1")
    cookie = (headers.get("Set-Cookie") or "").split(";")[0]
    req = urllib.request.Request(server + "/api/session", headers={"Cookie": cookie})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert json.loads(r.read())["role"] == "ADMIN"
