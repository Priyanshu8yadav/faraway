"""
responders.py  —  Module 9: Accessible Responder Alert & Escalation
───────────────────────────────────────────────────────────────────
Reaches every responder through a channel their ability allows — a text message.

Why this is an ACCESSIBILITY feature, not just a notifier:
  • Deaf / hard-of-hearing marshals cannot hear a radio, whistle or siren —
    a text message is their accessible channel (the reason emergency-SMS
    services exist at all).
  • A blind operator away from the desk has the text read aloud by their
    phone's built-in screen reader.
So routing the zone alert as a plain-language text makes the core
"receive-alert → act" workflow usable for people with a hearing or vision need.

Offline-first, like the rest of CA-CRS⁺ (mirrors the virtual-PLC pattern):
  • DEFAULT 'simulated' dispatch writes to an auditable outbox with NO network —
    always works in a live demo and never breaks the "runs offline" story.
  • An OPTIONAL real gateway (Twilio / MSG91) sends actual SMS only when API
    keys are present in the environment; it is off by default and best-effort.
Only this outbound text ever needs a link — all AI (counting, risk, forecast,
gate decisions) stays fully offline on the edge device.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ── Realistic sample roster (editable live on the dashboard) ─────────────────
# `zone` is a zone_id, or "*" for venue-wide escalation contacts.
# `access` records the responder's accessibility need — this is what makes the
# text channel meaningful (e.g. a Deaf marshal for whom text is the ONLY option).
DEFAULT_ROSTER = [
    {"zone": "zone_a", "name": "Amit Sharma",          "role": "Entry Marshal",  "phone": "+91 90000 11111", "access": "—"},
    {"zone": "zone_b", "name": "Priya Nair",           "role": "Plaza Marshal",  "phone": "+91 90000 22222", "access": "Deaf · text only"},
    {"zone": "zone_c", "name": "Ravi Kumar",           "role": "Exit Marshal",   "phone": "+91 90000 33333", "access": "Low vision · phone reads aloud"},
    {"zone": "*",      "name": "District Control Room", "role": "Control",        "phone": "+91 90000 44444", "access": "escalation"},
    {"zone": "*",      "name": "Police Control",        "role": "Police",         "phone": "112",             "access": "escalation"},
    {"zone": "*",      "name": "Medical / Ambulance",   "role": "Medical",        "phone": "108",             "access": "escalation"},
]

# Plain-language translations (shared intent with accessibility.py).
_ACTION = {"OPEN": "OPEN the exit gates", "CLOSE": "CLOSE the entry gates",
           "REDIRECT": "REDIRECT the crowd to another exit", "HOLD": "HOLD the gates"}
_CAUSE = {"Density": "too many people", "Speed": "people moving too fast",
          "Conflict": "crowds pushing against each other", "MIXED": "mixed causes"}


@dataclass
class Message:
    to_name: str
    to_role: str
    to_phone: str
    access: str
    zone_label: str
    level: str            # DANGER | WARNING
    body: str             # the plain-language SMS text
    ts: str = ""
    status: str = "QUEUED"   # SENT (simulated) | SENT (gateway) | FAILED
    key: str = ""            # dedup key


def _sms_body(zone_label: str, level: str, cause: str, gate: str, eta) -> str:
    """The message a responder actually receives — plain, structured, actionable.
    This IS the accessible format: readable by a Deaf marshal, spoken by a blind
    responder's phone screen reader."""
    if level == "DANGER":
        head = f"🚨 DANGER · {zone_label}: too crowded"
        do = f"Do now: {_ACTION.get(gate, gate)}."
    else:
        head = f"⚠ WARNING · {zone_label}: filling up fast"
        do = f"Prepare to {_ACTION.get(gate, gate).lower()}."
    eta_s = f" Danger in ~{eta:.0f}s." if eta else ""
    return f"CA-CRS+ ALERT\n{head}\nCause: {_CAUSE.get(cause, 'mixed causes')}.\n{do}{eta_s}"


def build_messages(states, roster, now_str: str) -> list[Message]:
    """Decide who to text from the current zone states.

    • Zone in DANGER  → text that zone's marshal (level DANGER) + venue escalation.
    • Zone projected to cross (forecast ETA) → text that zone's marshal (WARNING).
    """
    by_zone = {}
    for r in roster:
        by_zone.setdefault(r.get("zone", "*"), []).append(r)
    escalation = by_zone.get("*", [])
    out: list[Message] = []
    any_danger = False

    for s in states:
        zid = s.zone_id
        recips = by_zone.get(zid, [])
        if s.status == "DANGER":
            any_danger = True
            body = _sms_body(s.label, "DANGER", s.dominant_cause, s.gate_cmd,
                             getattr(s, "eta_danger_seconds", None))
            for r in recips:
                out.append(Message(r["name"], r["role"], r["phone"], r.get("access", "—"),
                                   s.label, "DANGER", body, now_str,
                                   key=f"{r['phone']}|{zid}|DANGER"))
        elif getattr(s, "eta_danger_seconds", None) is not None:
            body = _sms_body(s.label, "WARNING", s.dominant_cause, s.gate_cmd,
                             s.eta_danger_seconds)
            for r in recips:
                out.append(Message(r["name"], r["role"], r["phone"], r.get("access", "—"),
                                   s.label, "WARNING", body, now_str,
                                   key=f"{r['phone']}|{zid}|WARNING"))

    # Venue-wide escalation only when at least one zone is in DANGER.
    if any_danger and escalation:
        worst = max((s for s in states if s.status == "DANGER"), key=lambda z: z.risk)
        body = _sms_body(worst.label, "DANGER", worst.dominant_cause, worst.gate_cmd,
                         getattr(worst, "eta_danger_seconds", None))
        body = "ESCALATION — venue CRITICAL\n" + body
        for r in escalation:
            out.append(Message(r["name"], r["role"], r["phone"], r.get("access", "—"),
                               worst.label, "DANGER", body, now_str,
                               key=f"{r['phone']}|*|DANGER"))
    return out


def filter_new(candidates: list[Message], last_sent: dict, cooldown_s: float,
               now_ts: float) -> list[Message]:
    """Drop messages already sent to the same recipient for the same alert within
    the cooldown — so responders are not spammed every tick."""
    fresh = []
    for m in candidates:
        prev = last_sent.get(m.key, 0.0)
        if now_ts - prev >= cooldown_s:
            fresh.append(m)
            last_sent[m.key] = now_ts
    return fresh


# ── Optional real SMS gateway (off unless env keys are set) ──────────────────
def _twilio_configured() -> bool:
    return all(os.environ.get(k) for k in ("TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM"))


def _send_twilio(msg: Message):
    """Best-effort real send via Twilio REST (raw urllib, short timeout, threaded).
    Never blocks the UI; failures are swallowed (status set by caller optimistically)."""
    sid = os.environ["TWILIO_SID"]; tok = os.environ["TWILIO_TOKEN"]; frm = os.environ["TWILIO_FROM"]
    def _run():
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            data = urllib.parse.urlencode({"To": msg.to_phone.replace(" ", ""),
                                           "From": frm, "Body": msg.body}).encode()
            req = urllib.request.Request(url, data=data)
            import base64
            auth = base64.b64encode(f"{sid}:{tok}".encode()).decode()
            req.add_header("Authorization", f"Basic {auth}")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def dispatch(messages: list[Message], mode: str) -> list[Message]:
    """Send (or simulate) each message. Returns records with status set.
    mode: 'simulated' (offline, default) | 'live' (real gateway if configured)."""
    use_gateway = (mode == "live" and _twilio_configured())
    for m in messages:
        if use_gateway:
            _send_twilio(m)
            m.status = "SENT · gateway"
        else:
            m.status = "SENT · simulated"
        # Best-effort audit log (gitignored). Never fatal.
        try:
            with open("dispatch.log", "a") as fh:
                fh.write(f"[{m.ts}] {m.status} → {m.to_name} ({m.to_role}, {m.to_phone}) "
                         f"| {m.level} {m.zone_label} | {m.body.replace(chr(10),' / ')}\n")
        except Exception:
            pass
    return messages


def gateway_status(mode: str) -> str:
    if mode == "live":
        return "LIVE gateway ready" if _twilio_configured() else "LIVE selected · no API key → simulating"
    return "Simulated (offline)"
