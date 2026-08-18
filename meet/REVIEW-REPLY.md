# GWM review response kit

## Rejection round 3 (2026-08-18) — Bot Policy & Google Meet Media API (Developer Preview)

Reviewer findings:
- "The add-on must not provide functionality to invite a bot or rely on the presence of a bot in the meeting"
- Additional notes / screenshot of the "Protect this meeting" / "Deploy bot" screen: https://photos.app.goo.gl/aSvDGo9hAwhP9cuY8

**Status & Action Plan:**
1. **Google Workspace Developer Preview Program:** Application submitted for Project `940532414120` to gain access to the **Google Meet Media API**.
2. **Code (shipped):**
   - Side panel (`meet-addon.html`) stripped of all legacy bot participant language, "Admit bot" cards, and bot-specific states.
   - Replaced with native session initialization and real-time voice verification states.
3. **Backend Integration:**
   - Leveraging Recall.ai's "Meeting Direct Connect" wrapper for Google Meet Media API (streams real-time per-participant audio directly via WebRTC to `/api/ws/audio` without deploying a participant bot into the call).
4. **Marketplace resubmission:** Once Google approves the Developer Preview enrollment, enable Google Meet Media API in Cloud Console, add required media scopes (`meetings.conference.media.readonly`), and resubmit.

---

## Rejection round 2 (2026-08-14 3:22 PM) — fixes + YOUR SDK steps

Reviewer findings: (1) pricing model not set; (2) "add-on must provide meeting
functionality — after login the user sees this screen only with no visible
changes." Admin-feed forensics: the reviewer signed in successfully twice
(14:35 / 14:42 ET — round-1 auth fix confirmed working) but never deployed a
bot; the static protect screen didn't demonstrate function to them.

**Code (shipped):**
- The protect screen now offers "▶ Watch a 15-second simulated detection" —
  clearly-labeled sample data showing the full live monitoring UI (speaker
  cards, verdict flipping to FAKE, red wire-hold styling, session stats) with
  zero setup: no second participant, no bot admission. A reviewer sees the
  meeting functionality in one click.
- If the Meet SDK can't supply the meeting code, the panel reveals a
  paste-the-meeting-link input instead of dead-ending — Protect always works.

**Your steps in the Marketplace SDK before republishing:**
1. **App Listing → Pricing**: select **"Paid with free features"**
   (free 5 monitored hours/month, then $8/monitored-hour).
2. **Screenshots**: upload the refreshed set from `designs/marketplace/` and
   order them to match the real first-run experience:
   `shot-protect.png` FIRST, then `shot-meet-panel.png`, then
   `shot-wire-hold.png` (+ console/landing shots as desired).
3. Republish.

Optional reply in the rejection thread:

> Hi — both points are addressed: (1) the pricing model is now set to "Paid
> with free features" and shows on the listing; (2) the side panel now
> demonstrates its meeting functionality immediately — the first screen offers
> a clearly-labeled 15-second simulated detection preview (speaker cards and a
> live deepfake verdict) requiring no setup, and one-click "Protect this
> meeting" deploys the scoring bot into the call (with a paste-the-link
> fallback). Listing screenshots were re-shot to match the actual first-run
> screens in order. Republished and live.



## Rejection round 1 (2026-08-14) — FIXED, ready to republish

Reviewer findings and what changed (shipped in the `gis` auth rebuild):

1. **"After Signin nothing happens" / must work with 3rd-party cookies
   disabled** → the popup + postMessage flow is GONE. The panel now uses
   **Google Identity Services**: One Tap prompt (FedCM,
   `use_fedcm_for_prompt`) with the **official rendered Sign in with Google
   button** as fallback. The GIS ID token is verified server-side
   (`/auth/google-credential`, audience/issuer/expiry checked) and swapped
   for a Sonave session token held in partitioned localStorage — zero
   cookies involved, works with third-party cookies fully disabled,
   sign-in happens once per browser (auto_select re-signs silently).
2. **Logout must revoke tokens** → the panel has a **Sign out** link;
   logout now bumps the server-side session version, instantly revoking
   every outstanding Sonave session token (covered by test:
   `test_logout_revokes_all_session_tokens`), calls
   `google.accounts.id.disableAutoSelect()`, and lands on the sign-in
   screen with One Tap re-armed.

**To resubmit**: Marketplace SDK → Publish (republish the same listing).
Optionally reply in the rejection email thread:

> Hi — both findings are addressed: the add-on now signs in via Google
> Identity Services (One Tap with FedCM + the official Sign in with Google
> button; no popup, functional with third-party cookies disabled), and
> Sign out revokes all session tokens server-side and returns to the
> sign-in screen. The updated version is republished and live at the same
> deployment. Thank you for the specific repro video — it pointed straight
> at the fix.



Google started the Marketplace review (2026-08-14). They asked for: testing
credentials (if any), allowlisting of `gsmtestuser@marketplacetest.net`, and a
screen recording of the end-to-end workflow showing scope usage.

## Before replying — 2-minute checklist

1. **GCP → APIs & Services → OAuth consent screen**: confirm publishing status
   is **In production** (it should be — brand verification is pending on it).
   If it ever shows **Testing**, add `gsmtestuser@marketplacetest.net` under
   *Test users*, or sign-in will be blocked for the reviewer regardless of our
   open signup.
2. Optionally add `gsmtestuser@marketplacetest.net` as an OAuth test user
   anyway — harmless, and lets you truthfully say it's allowlisted.
3. Record the demo video (script below), upload **unlisted to YouTube** or
   Drive (link-visible), paste the link into the reply.

## Reply email (paste to gwm-review@google.com, reply in-thread)

---

Hi GWM Review Team,

Thank you for starting the review of Sonave (project 940532414120).

**Testing credentials:** none are required. Sign-up is open and self-serve —
the reviewer can sign in with any Google account, including
gsmtestuser@marketplacetest.net. Nothing needs to be allowlisted on our side.
Every new account automatically receives 5 free monitored hours, which is more
than sufficient for functionality testing; no payment method is required.

**End-to-end test flow** (also shown in the recording below):

1. In a Google Meet call, open Activities → **Sonave** — the side panel opens.
2. Click **Continue with Google**. The OAuth popup requests only basic
   identity scopes (openid, email, profile) and closes itself after sign-in.
3. Click **Protect this meeting** — our notetaker-style bot ("Sonave") asks to
   join the call. Admit it from Meet's prompt (or People → Waiting to join).
   The panel narrates the bot's status while it joins.
4. Speak normally: each speaker gets a live card with a voice-authenticity
   verdict in ~4 seconds (REAL / SUSPECT / FAKE with a confidence percentage),
   refreshed every 4 seconds. The signed-in user's own card is tagged "· you".
5. **Open console** links to the full dashboard at
   https://usesonave.com/console (live monitor, session history, forensic
   incident reports, billing).

**Scope usage:** Sonave requests only openid, email and profile. They are used
solely to create and identify the user's workspace, display the signed-in
identity, and tag the user's own speaker card in the panel. No sensitive or
restricted scopes are requested, and no Google user data is stored beyond the
account identity (subject id, email, name, picture).

**Screen recording:** [VIDEO LINK]

Privacy policy: https://usesonave.com/privacy · Terms of service:
https://usesonave.com/terms

Please let us know if you need anything else.

Best regards,
Derek Gallardo
Sonave — https://usesonave.com

---

## Demo video script (~2 minutes, one take, screen + mic optional)

Record the browser at a comfortable zoom. No editing needed beyond trimming.

1. **(0:00)** Start in a Google Meet call you host. Open **Activities →
   Sonave**. The panel shows the sign-in screen.
2. **(0:10)** Click **Continue with Google** — let the OAuth popup render a
   beat so the requested scopes (name, email, profile picture) are visible on
   screen, then complete sign-in. Popup closes itself; panel switches to the
   protect screen.
3. **(0:30)** Read the three steps aloud or hover them briefly. Click
   **Protect this meeting**. Show the "Bot joining this meeting…" card with
   its live status line.
4. **(0:45)** Admit **Sonave** from Meet's prompt. Point out the bot appears
   as a visible participant — everyone in the call can see it.
5. **(0:55)** Talk for ~15 seconds. Show: your card appears tagged "· you",
   the equalizer animates while you speak, the first verdict lands in ~4 s,
   the percentage and REAL badge update live, CHECKS/SESSION stats tick.
6. **(1:15)** Mute yourself for ~5 s → card dims to "muted / quiet". Unmute.
7. **(1:25)** Click **Open console** — show the same speaker live in the
   console's monitor, then History with past sessions and an incident report
   opening (any past incident → Export report).
8. **(1:45)** Back in Meet: remove the bot from the call. Show the panel
   returning to the protect screen by itself. End.

Covers: both OAuth touchpoints and what the scopes power ("· you", account
chip in console), the full user workflow, billing-free testing, and graceful
lifecycle — everything their email asked to see.
