# CASA AL1 — remediation & hardening ledger (assessment window)

Internal. Every security change shipped after the evidence pack was submitted,
mapped to CASA requirement IDs. If the assessor raises any of these rows, the
response is below, verbatim, with commit references — all changes are live at
https://usesonave.com and re-verifiable.

| Date (2026) | Req IDs | Change | Commit / proof |
|---|---|---|---|
| 09-03 | **4.1.1** (row 27) | Cloudflare Minimum TLS Version raised to **1.2**. Explicit per-version handshake probe now shows TLS 1.0/1.1 rejected with protocol-version alerts, 1.2/1.3 negotiated. | Cloudflare dashboard setting; probe script `scratch/tls_probe` output archived in session log |
| 09-03 | **5.1.5** (row 35) | SSRF guard `railway/netsafe.py`: user-supplied URLs (secret iCal feeds, alert webhooks) restricted to public-HTTPS — loopback, RFC1918, link-local/cloud-metadata and reserved ranges rejected at DNS-resolution level; redirects refused; enforced at save time (422) AND fetch time. 8 unit tests. | `e56140a` |
| 09-02 | **2.4.1, 3.1.1, 3.1.3** (rows 17/18/20) | Closed anonymous surface introduced by recent feature work: training/HF endpoints now require auth (admin for start/schedule); HF webhook secret-gated fail-closed (was 500ing); demo websocket auth repaired (broken principal call had dropped anonymous callers into the admin workspace). New permanent regression net: `tests/test_route_posture.py` sweeps EVERY route anonymously — public pages must 200, everything else must never 200 or 5xx; websockets must close 1008. | `d4f4edb` |
| 09-03 | **2.1.1** (row 9) | Token-in-URL eliminated on HTTP: `get_principal` no longer accepts `?token=` query params (headers + cookies only); the console's `?tok=` bootstrap and the panel's token-bearing console link removed. Residual (documented): per-bot WebSocket tokens ride the WS URL because Recall's realtime endpoints support no header channel — they are bot-scoped, 24 h-expiring, and stored hashed; uvicorn logs WS paths without query strings. | this commit |
| 09-03 | **2.3.1, 2.3.2** (rows 13/14) | Operator token cookie is now set **server-side via `POST /auth/operator`** with `HttpOnly; Secure; SameSite=Strict` — JavaScript never reads or writes it; logout deletes it server-side. | this commit |
| 09-03 | **1.1.1** (row 1) | Rate limiter: dropped trust in caller-spoofable `X-Forwarded-For` (Cloudflare's `cf-connecting-ip` or the direct peer only) and added test coverage: 25-request burst → 20 processed, 21+ → 429 with `Retry-After: 60` (`tests/test_rate_limit.py`). | this commit |
| 09-03 | **6.5.1** (row 46) | Login log lines now mask emails (`d***@gmail.com`) — over-compliance; no credentials/tokens/payment data were ever logged. | this commit |
| 09-03 | integrity | Simulated mic-morph demo made honest and operator-only: admin-gated websocket + hidden card for non-admins; forced speaker label "Simulated Mic Test"; model field `"simulation"`; invented vendor attribution replaced with "simulated demo signal". | this commit |

Standing evidence available on request: 2-Step Verification screenshot for the
admin Google account (row 26); `pip-audit` clean run dated 09-02 (row 42);
route-posture and tenancy-isolation test suites (rows 17–21).
