# Competitive brief — voice-deepfake detection for meetings

Internal. For sales calls, positioning decisions, and objection handling.
Last updated 2026-08-15 (Resemble DETECT-World launch week).

## Landscape

| Vendor | What they sell | Meeting story | Accuracy marketing | Pricing posture |
|---|---|---|---|---|
| **Resemble AI** (Detect / DETECT-World) | Multi-modal detection (audio/video/image), API + SDKs, Chrome ext, desktop app, enterprise integrations (Salesforce, Workday, Okta, contact-center stacks) | Bots for Meet/Zoom/Teams/Webex, calendar auto-join — launched broad, depth unproven | "Up to 99.5% across modalities", "250+ generators", "#1 on third-party benchmarks" | Free API teaser → contact sales; card-gated workspace |
| **Pindrop** | Call-center voice security incumbent (phone channel), now "Pindrop Pulse" for meetings | Meetings product newer; DNA is telephony | Enterprise claims, no public numbers | ~$45/user/mo leaked; enterprise sales only |
| **Reality Defender** | Multi-modal detection API/platform, deepfake screening | No native in-meeting bot story (API/file oriented) | High headline accuracy claims | Enterprise sales |

**Sonave**: real-time per-speaker verdicts inside the meeting, trained specifically
on meeting-codec audio, wired to a wire-hold webhook + forensic report. Public
self-serve pricing (free 5 h/mo → $8/monitored-hour). Published benchmarks with
the weak numbers included.

## Our wedge (in order of strength)

1. **Meeting reality vs lab accuracy.** Same 27 unseen commercial voice tools,
   played through real meeting audio: commodity detector 1.9% catch, Sonave
   95.2%. Lab numbers evaporate on Opus-compressed, processed call audio —
   that is the entire reason Sonave exists. (results/benchmark_baseline.json)
2. **The verdict does something.** Wire-hold webhook into the approval flow +
   3-consecutive-window discipline + exportable forensic report. Everyone else
   stops at a score in a dashboard.
3. **Radical benchmark honesty.** We publish ~59% on In-the-Wild and say
   "second factor, not a replacement." Fraud buyers have been burned by 99%
   marketing; honesty is a moat that costs competitors their marketing dept.
4. **Self-serve speed + transparent price.** Sign in with Google, bot in your
   call in 60 seconds, $8/monitored-hour on the site. No demo call, no MSA
   before value.

## Objection handling

**"Resemble claims 99.5% — you claim 95% (and 59% on hard stuff). Aren't they better?"**
Those are different questions. 99.5% is a lab number across modalities on their
chosen sets; our 95% is measured on *unseen commercial voice tools played
through real meeting audio* — and on that same set, a well-known commodity
detector scores 2%. Ask any vendor for their number **through the meeting
codec, on tools absent from training** — and whether they'll publish it. We do:
usesonave.com/benchmarks.

**"They do video and images too."**
Wire fraud is a voice crime. The $25M Arup case, the CEO-voice transfers —
audio deepfakes on live calls. A video detector doesn't hold your wire. If you
need file-scanning across modalities, use one of those too — Sonave is the
in-call voice layer with the payment workflow.

**"They're bigger / 4.5M users."**
Bigger means enterprise sales cycles and contact-us pricing. You can protect
this afternoon's approval call with Sonave before their SDR replies. Free tier,
public price, cancel anytime.

**"They have on-prem / air-gapped."**
True, and if that's a hard requirement today we're not your vendor yet — honest
loss. Most wire-desks run SaaS approval stacks already; our data posture
(named subprocessors, deletable voiceprints, training opt-out) is published.

**"What about Pindrop?"**
Phone-channel DNA, per-seat enterprise pricing (~$45/user/mo leaked), sales
motion. Sonave is meetings-first, usage-priced, self-serve. Different lane;
if they're evaluating Pindrop, they have budget — sell the workflow depth.

## When we genuinely lose (do not fight these)

- Hard multi-modal requirement (video deepfake detection in-scope).
- Hard on-prem / air-gap requirement.
- Contact-center telephony scale (Pindrop's home turf).
- Procurement requires SOC 2 today (roadmap, not reality — do not claim it).

## Head-to-head bench status (decided 2026-08-15)

**Do not run our test sets through Resemble's API.** Their ToS §8 prohibits
accessing the service "for the purposes of monitoring its availability,
performance or functionality, or for any other benchmarking or competitive
purposes," and separately bars using outputs "to train, improve, or otherwise
further develop... deepfake detection model[s]." A published head-to-head from
our account would be a contract breach and a reputational gift to them.
Legitimate alternatives: (a) our open-source commodity comparator (published),
(b) citing independent academic evaluations of commercial detectors when they
exist, (c) prospects running their own procurement bake-offs — encourage
those; our /benchmarks page tells them exactly what to ask every vendor.

## Pricing comparison one-liner

"Free for 5 monitored hours a month, then $8 per monitored hour, on the
website, no sales call. Ask them for a price and a real-world number; we've
published both."
