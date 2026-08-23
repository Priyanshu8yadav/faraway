"""
accessibility.py  —  Module 8: Accessible Command Channel
─────────────────────────────────────────────────────────
Round-2 challenge (#434): make the core operator workflow — perceive an alert
and know the action — usable for people with an accessibility need.

A crowd-safety alert must reach EVERY operator. The base dashboard encodes risk
almost entirely by colour (green/amber/red) — unusable for the ~1-in-12 men with
red–green colour-vision deficiency, and invisible to blind / low-vision staff.

This module rebuilds the alert as a REDUNDANT, MULTI-MODAL message so no single
sense or ability is required to receive it (WCAG 1.4.1 use-of-colour,
1.1.1 text alternatives, 4.1.3 status messages):

  1. Colour-blind-safe visual  — Okabe–Ito palette + a distinct SHAPE + WORD per
                                 status (never colour alone).
  2. Plain-language text        — "what's wrong · where · what to do · how urgent".
  3. Screen-reader announcement — an aria-live="assertive" role="alert" region.
  4. Spoken audio (TTS)         — the alert is read aloud.

Everything is captioned by design (the spoken text is always shown as text), so
the audio channel also serves deaf / hard-of-hearing operators.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import threading

# ── Okabe–Ito colour-blind-safe palette (distinguishable for all CVD types) ──
CB_PALETTE = {"SAFE": "#009E73", "WARNING": "#E69F00", "DANGER": "#D55E00"}
STD_PALETTE = {"SAFE": "#22c55e", "WARNING": "#f59e0b", "DANGER": "#ef4444"}
# Redundant, non-colour encodings — distinct silhouettes + plain words.
SHAPE = {"SAFE": "●", "WARNING": "▲", "DANGER": "■"}
WORD  = {"SAFE": "ALL CLEAR", "WARNING": "GETTING BUSY", "DANGER": "TOO CROWDED"}

# Plain-language translations (no jargon, no formulas).
_ACTION = {
    "OPEN":     "Open the exit gates",
    "CLOSE":    "Close the entry gates",
    "REDIRECT": "Redirect the crowd to another exit",
    "HOLD":     "Hold — keep the gates as they are",
}
_CAUSE = {
    "Density":  "too many people",
    "Speed":    "people moving too fast",
    "Conflict": "crowds pushing against each other",
    "MIXED":    "mixed causes",
}


def status_style(status: str, colorblind: bool):
    """Return (color, shape, word) for a status — colour is never the only cue."""
    pal = CB_PALETTE if colorblind else STD_PALETTE
    return pal.get(status, "#94a3b8"), SHAPE.get(status, "●"), WORD.get(status, status)


def build_alert(states) -> dict:
    """Collapse all zones into ONE highest-priority, plain-language alert.

    Priority:  any DANGER  >  a forecast projecting DANGER  >  all clear.
    Returns a dict with level, the on-screen text, and the spoken text.
    """
    dangers = [s for s in states if s.status == "DANGER"]
    if dangers:
        s = max(dangers, key=lambda z: z.risk)
        action = _ACTION.get(s.gate_cmd, s.gate_cmd)
        cause = _CAUSE.get(s.dominant_cause, "mixed causes")
        eta = ""
        if getattr(s, "eta_danger_seconds", None):
            eta = f" Danger in about {s.eta_danger_seconds:.0f} seconds."
        return {
            "level": "DANGER", "zone": s.label,
            "headline": f"{s.label} is too crowded",
            "detail": f"Cause: {cause}. Action: {action}.{eta}",
            "speak": f"Alert. {s.label} is too crowded. {action}.{eta}",
        }
    rising = [s for s in states if getattr(s, "eta_danger_seconds", None) is not None]
    if rising:
        s = min(rising, key=lambda z: z.eta_danger_seconds)
        action = _ACTION.get(s.gate_cmd, "prepare to act")
        return {
            "level": "WARNING", "zone": s.label,
            "headline": f"{s.label} is filling up fast",
            "detail": f"Projected danger in about {s.eta_danger_seconds:.0f} seconds. Prepare to {action.lower()}.",
            "speak": f"Warning. {s.label} is filling up. Danger projected in about {s.eta_danger_seconds:.0f} seconds.",
        }
    return {"level": "SAFE", "zone": "", "headline": "All zones clear",
            "detail": "No action needed.", "speak": None}


def alert_token(alert: dict) -> str:
    """Stable id for an alert so audio/announcements fire ONCE per change."""
    return f"{alert['level']}|{alert['zone']}|{alert['headline']}|{alert['detail']}"


def accessible_panel_html(alert: dict, colorblind: bool, large: bool) -> str:
    """The Accessible Command Panel — high-contrast, shape+word+colour, plain text.
    Carries role/aria so assistive tech treats it as a live status region too."""
    color, shape, word = status_style(alert["level"], colorblind)
    fs_head = "34px" if large else "26px"
    fs_body = "20px" if large else "15px"
    pad = "22px 26px" if large else "16px 20px"
    # High-contrast: near-black text on the status colour for WARNING/DANGER;
    # white text works on all three Okabe–Ito tones at these sizes.
    return (
        f"<div role='status' aria-live='polite' "
        f"style='background:{color};border-radius:14px;padding:{pad};margin-bottom:14px;"
        f"border:3px solid rgba(255,255,255,0.35);'>"
        f"<div style='display:flex;align-items:center;gap:16px'>"
        f"<span aria-hidden='true' style='font-size:{fs_head};line-height:1;color:#fff'>{shape}</span>"
        f"<div>"
        f"<div style='font-size:{fs_head};font-weight:800;color:#fff;line-height:1.05'>"
        f"{word} &nbsp;—&nbsp; {alert['headline']}</div>"
        f"<div style='font-size:{fs_body};font-weight:600;color:#fff;opacity:0.95;margin-top:4px'>"
        f"{alert['detail']}</div>"
        f"</div></div></div>"
    )


def announce_component_html(alert: dict, token: str, speak: bool, rate: float = 1.0) -> str:
    """HTML+JS for a hidden aria-live region that screen readers announce, plus
    optional in-browser Text-to-Speech (Web Speech API — offline, no service).
    Re-render this only when `token` changes so it announces/speaks ONCE."""
    say = alert.get("speak")
    speak_js = ""
    if speak and say:
        speak_js = (
            "try{var u=new SpeechSynthesisUtterance(" + json.dumps(say) + ");"
            f"u.rate={float(rate):.2f};u.lang='en-US';"
            "window.speechSynthesis.cancel();window.speechSynthesis.speak(u);}catch(e){}"
        )
    live_text = f"{alert['headline']}. {alert['detail']}"
    return (
        "<div aria-live='assertive' role='alert' "
        "style='position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)'>"
        f"{live_text}</div>"
        f"<script>/*{token}*/{speak_js}</script>"
    )


# ── Server-side spoken alert (macOS `say`) — reliable audio for a live demo ──
_HAS_SAY = shutil.which("say") is not None

def speak_native(text: str):
    """Speak `text` out loud on THIS machine via macOS `say`, off the UI thread.
    No-op where `say` is unavailable. Best-effort; never raises."""
    if not text or not _HAS_SAY:
        return
    def _run():
        try:
            subprocess.Popen(["say", text])
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()
