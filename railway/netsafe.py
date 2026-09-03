"""Outbound-request safety (SSRF guard, CASA/ASVS 5.1.5).

User-supplied URLs that the SERVER later fetches — secret iCal feeds and alert
webhooks — must never let a user point us at private infrastructure. Enforced
both when a URL is saved (clear error to the user) and when it is fetched
(defense in depth if a stored URL's DNS changes).

Known limitation, acceptable at this assurance level: resolution happens
before the fetch (classic TOCTOU/DNS-rebinding window). Redirect-following is
disabled separately by callers so a public URL cannot bounce us private.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request


def assert_public_https(url: str) -> None:
    """Raise ValueError unless url is https to a host resolving ONLY to
    public addresses."""
    u = urllib.parse.urlparse(url)
    if u.scheme != "https":
        raise ValueError("only https:// URLs are allowed")
    host = u.hostname or ""
    if not host:
        raise ValueError("invalid URL")
    try:
        infos = socket.getaddrinfo(host, u.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ValueError("unresolvable host")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError("URL resolves to a non-public address")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D102 — refuse all redirects
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def open_public(url: str, data: bytes | None = None, headers: dict | None = None,
                timeout: float = 15):
    """urlopen for user-supplied URLs: public-https enforced, redirects refused."""
    assert_public_https(url)
    req = urllib.request.Request(url, data=data, headers=headers or {})
    return _OPENER.open(req, timeout=timeout)
