# GWM review response kit

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
