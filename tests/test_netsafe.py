"""SSRF guard (netsafe): user-supplied URLs must never reach private networks."""
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "railway"))
import netsafe  # noqa: E402


@pytest.mark.parametrize("url", [
    "http://example.com/a.ics",                 # not https
    "https://127.0.0.1/a.ics",                  # loopback
    "https://localhost/a.ics",                  # loopback by name
    "https://10.0.0.5/hook",                    # RFC1918
    "https://192.168.1.1/hook",                 # RFC1918
    "https://169.254.169.254/latest/meta-data", # link-local / cloud metadata
    "https://[::1]/a.ics",                      # v6 loopback
    "ftp://example.com/x",                      # wrong scheme
])
def test_rejects_private_and_non_https(url):
    with pytest.raises(ValueError):
        netsafe.assert_public_https(url)


def test_accepts_public_host(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("142.250.80.14", 443))])
    netsafe.assert_public_https("https://calendar.google.com/calendar/ical/x/basic.ics")


def test_rejects_host_resolving_private(monkeypatch):
    """DNS pointing a pretty hostname at an internal address must be refused."""
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("10.1.2.3", 443))])
    with pytest.raises(ValueError):
        netsafe.assert_public_https("https://innocent-looking.example.com/feed.ics")


def test_open_public_refuses_redirects(monkeypatch):
    """A public URL must not be able to bounce the server elsewhere via 3xx."""
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("142.250.80.14", 443))])
    seen = {}

    class _Resp:
        def __init__(self):
            self.status = 302

        def read(self):
            return b""

    def _open(self, req, timeout=None):
        seen["redirects_disabled"] = any(
            isinstance(h, netsafe._NoRedirect) for h in self.handlers)
        raise OSError("stop before network")

    monkeypatch.setattr(type(netsafe._OPENER), "open", _open)
    with pytest.raises(OSError):
        netsafe.open_public("https://calendar.google.com/x.ics")
    assert seen["redirects_disabled"] is True
