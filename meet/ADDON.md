# Sonave — Google Meet Add-on: Deployment & Marketplace Submission Runbook

Everything needed to put the Sonave side panel inside Google Meet and submit it
to the Workspace Marketplace. The panel itself is live at
`https://usesonave.com/meet-addon` (works standalone; inside Meet it binds via
the Add-ons SDK). Prereq: the OAuth **brand verification must be approved**
before the public Marketplace listing (unlisted testing works before that).

## 1. Enable the APIs (Google Cloud console, Sonave project)

APIs & Services → Library → enable:
- **Google Workspace Marketplace SDK**
- **Google Workspace Add-ons API**

## 2. Create the HTTP deployment

Marketplace SDK → **HTTP deployments** tab → **Create new deployment**:
- Deployment name: `sonave-meet`
- Paste the contents of [`deployment.json`](deployment.json) (in this folder)
  into the DEPLOYMENT.JSON panel → Save.

Then note the project **number** (console dashboard, not the project id) and set
it on Railway: `SONAVE_MEET_PROJECT_NUMBER=<number>` — the panel passes it to
`createAddonSession()`.

## 3. App configuration (Marketplace SDK → App configuration tab)

- App integration: **Web app** → *Deploy using cloud deployment resource* →
  select the `sonave-meet` deployment.
- App visibility: **Public**, listing **Unlisted** first (flip to listed after
  testing + verification).
- OAuth scopes: `openid`, `.../auth/userinfo.email`, `.../auth/userinfo.profile`
  (must exactly match the consent screen's Data Access list).
- Developer info: name **Sonave**, email derekgallardo01@gmail.com.

## 4. Test it yourself before submitting

1. HTTP deployments tab → **Install** (Actions column) — installs for your account.
2. Start a meeting at meet.google.com → **Activities** (triangle/square/circle
   icon) → Sonave → the side panel loads `usesonave.com/meet-addon`.
3. Sign in via the popup button (the panel stores a partitioned session cookie
   that works in Meet's iframe), deploy a bot into the same meeting from the
   console, talk — the first verdict appears after ~4-5 s of actual speech.

## 5. Store listing (Marketplace SDK → Store listing tab) — ready to paste

| Field | Value |
|---|---|
| App name | `Sonave` |
| Short description (≤120 chars) | `Live deepfake-voice detection inside your meetings — every speaker gets a REAL / SUSPECT / FAKE verdict.` |
| Detailed description | `Sonave watches the voices in your meeting and tells you, in real time, whether each one is human. A visible Sonave bot joins the call and streams per-speaker audio to a detection model trained on real meeting-codec audio; the side panel shows a live authenticity meter per speaker and a room-level verdict. When a voice scores in the red band for three consecutive windows, Sonave raises a wire-hold incident — with a webhook that can pause a payment approval and a one-click forensic report for compliance. Built for finance teams approving wires on calls, and for anyone who needs to know the voice on the other end is real. Free tier: 5 monitored hours per month. Then $8 per monitored hour. Enterprise: usesonave.com.` |
| Category | Productivity (or Business tools) |
| App icon 128×128 | `designs/marketplace` ← use `https://usesonave.com/icon-128.png` / upload the file |
| Screenshots 1280×800 (upload in this order) | `designs/marketplace/shot-meet-panel.png` · `shot-wire-hold.png` · `shot-console.png` · `shot-landing.png` |
| Homepage / support URL | `https://usesonave.com` |
| Privacy policy | `https://usesonave.com/privacy` |
| Terms of service | `https://usesonave.com/terms` |
| Support email | `derekgallardo01@gmail.com` |
| Regions | All regions |
| Pricing | Free with paid features |

## 6. Submit for review

Store listing tab → **Publish** → submits to the Marketplace review queue.
Reviews commonly ask for: a demo video or test instructions (point them at the
open free signup: "sign in at usesonave.com/console with any Google account"),
and confirmation the OAuth scopes match the consent screen. Expect days to a
couple of weeks; the unlisted install keeps working throughout.

## Notes / gotchas

- `addOnOrigins` in deployment.json must exactly cover where the panel is
  served (`https://usesonave.com`). If the domain ever changes, update BOTH the
  deployment JSON and `SONAVE_MEET_PROJECT_NUMBER`-adjacent Railway vars.
- The panel authenticates in the iframe via a `Partitioned; SameSite=None`
  companion cookie set at sign-in (CHIPS). If a browser blocks it, the panel
  falls back to the popup sign-in button.
- Do NOT add Calendar scopes to this listing until the calendar auto-join
  feature ships — scope lists must match the consent screen exactly.
