# Sonave — Secrets Management Policy

**Owner:** Derek Gallardo (sole operator) · **Effective:** 2026-09-04 · **Review:** quarterly, and
immediately after any suspected exposure.

## 1. Inventory

| Secret | Purpose | Storage location |
|---|---|---|
| `SONAVE_SESSION_SECRET` | HMAC-SHA256 session signing key (256-bit) | Railway env var |
| `SONAVE_API_TOKEN` | Operator/machine token | Railway env var |
| `SONAVE_GOOGLE_CLIENT_SECRET` | Google OAuth client secret | Railway env var |
| Stripe secret + webhook signing keys | Billing | Railway env var |
| Recall.ai / Modal / Hugging Face tokens | Bot deploy, GPU scorer, model hub | Railway env vars; CI copies in GitHub Actions encrypted secrets |
| User Google OAuth refresh tokens | Calendar auto-join (flag-gated) | Application DB on Railway encrypted persistent volume |
| Per-bot stream tokens, user API keys | Audio ingest, API access | **SHA-256 digest only** in DB; plaintext shown once at creation, never stored |

## 2. Storage rules

- Secrets exist **only** in Railway environment variables (encrypted at rest by the platform) or
  GitHub Actions encrypted secrets. Never in source code, git history, container images, logs, or
  client-side code. `.env` (local dev) is git-ignored.
- Application code reads secrets exclusively via `os.environ.get(...)` at startup.
- Derived credentials (bot tokens, API keys) are persisted only as irreversible SHA-256 hashes.

## 3. Access control

- One human principal (the operator) on each control plane: Railway project, GitHub repository,
  Cloudflare account, Google Cloud console, Stripe account. No shared or service accounts with
  dashboard access.
- Every operator account enforces 2-Step Verification (FIDO2 security key / passkey).
- Production runtime is the only non-human reader; CI receives only the deploy-scoped tokens it
  needs (`MODAL_TOKEN_*`, `SONAVE_API_TOKEN`, `HF_TOKEN`) via `${{ secrets.* }}` references.

## 4. Rotation & revocation

- **Routine rotation:** third-party keys (Stripe, Recall, Modal, HF, Google client secret) are
  rotated via their consoles and updated in Railway; the service picks them up on redeploy.
- **Session signing secret:** rotating `SONAVE_SESSION_SECRET` invalidates every outstanding
  session token at once (accepted behavior — users re-authenticate via Google).
- **Targeted revocation:** per-user `session_ver` bump invalidates all of one user's sessions
  immediately; Google OAuth tokens are revoked at `oauth2.googleapis.com/revoke` on logout or
  disconnect; user API keys and bot tokens are deleted by row (hash-stored, so deletion is final).
- **On suspected exposure (in order, same day):** rotate the secret at its source → update Railway
  env var → redeploy → global `session_ver` bump if session-adjacent → review Railway/Cloudflare
  activity logs and the application audit trail for misuse → record the incident and remediation
  in `docs/casa-addendum.md`.

## 5. Monitoring

- Railway project activity log records configuration and deployment access (operator-reviewed).
- The application audit trail (`events` table) records every sign-in, sign-out, session
  revocation, key creation/deletion, and bot lifecycle event, and pushes each to the operator in
  real time (admin dashboard + email/Slack).
- CI is the only automated writer to production and installs from the single pinned requirements
  source; dependency health is checked with `pip-audit`.
