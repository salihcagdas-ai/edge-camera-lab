"""Local web UI + API (FastAPI, served by uvicorn from ``edge.__main__``).

Endpoints: ``/`` (UI), ``/stream`` (MJPEG live view — also the remote
cat-watching path over Tailscale), ``/events``, ``/mode``, ``/people``
(list + DELETE), ``/enroll`` (photo upload), ``/snapshots/*``, ``/clips/*``
(event videos), ``/health``.

UI strings are Turkish (owner-facing); code stays English.

Auth: a single owner PIN gates every data route (live stream, snapshots,
clips, events, mode, enroll, people). It is a cookie session signed with a
per-process secret, so ``<img src="/stream">`` and snapshot links carry it
automatically. An empty ``web_pin`` disables auth (dev/tests only). Tailscale
gates REMOTE reach; the PIN additionally protects same-LAN peers (a housemate
or Wi-Fi guest). See docs/privacy.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edge.config import EdgeSettings
from edge.embedder import FaceEmbedder
from edge.hub import FrameHub
from edge.matcher import MatcherRef
from edge.presence import MODE_AWAY, MODE_HOME, PresenceState
from edge.store import HomeStore

_COOKIE = "edge_camera_session"
_SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days
_LOGIN_WINDOW = 300.0  # brute-force throttle window (s)
_LOGIN_MAX_FAILURES = 5  # failed /login attempts per IP before lockout
_MAX_UPLOAD_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_UPLOAD_FILE_BYTES = 8 * 1024 * 1024
_MAX_UPLOAD_FILES = 10
_MAX_IMAGE_PIXELS = 20_000_000


class _UploadBodyLimit:
    """Bound enrollment bodies before multipart spooling, including chunked requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/enroll":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await JSONResponse({"detail": "invalid content length"}, status_code=400)(
                scope, receive, send
            )
            return
        if length < 0 or length > _MAX_UPLOAD_REQUEST_BYTES:
            await JSONResponse({"detail": "upload request too large"}, status_code=413)(
                scope, receive, send
            )
            return
        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > _MAX_UPLOAD_REQUEST_BYTES:
                    raise HTTPException(status_code=413, detail="upload request too large")
            return message

        await self.app(scope, limited_receive, send)


@dataclass
class AppState:
    settings: EdgeSettings
    store: HomeStore
    presence: PresenceState
    matcher_ref: MatcherRef
    embedder: FaceEmbedder
    hub: FrameHub
    decode_image: Callable[[bytes], np.ndarray | None]
    encode_jpeg: Callable[[np.ndarray], bytes | None]
    web_pin: str = ""
    session_secret: str = field(default_factory=lambda: secrets.token_hex(32))
    cookie_secure: bool = False
    session_epoch: int = 0  # bumped on logout to revoke outstanding tokens
    login_failures: dict[str, list[float]] = field(default_factory=dict)


class ModeRequest(BaseModel):
    mode: str


class PinRequest(BaseModel):
    pin: str = Field(min_length=1, max_length=128)


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _make_token(secret: str, epoch: int, now: float) -> str:
    payload = f"{int(now)}.{epoch}"
    return f"{payload}.{_sign(secret, payload)}"


def _valid_token(token: str | None, secret: str, epoch: int, now: float) -> bool:
    if not token or len(token) > 256:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    issued_str, epoch_str, sig = parts
    if len(sig) != 64 or not sig.isascii():
        return False
    if not hmac.compare_digest(sig, _sign(secret, f"{issued_str}.{epoch_str}")):
        return False
    try:
        issued, token_epoch = int(issued_str), int(epoch_str)
    except ValueError:
        return False
    if token_epoch != epoch:  # revoked by a logout
        return False
    return 0 <= (now - issued) <= _SESSION_MAX_AGE


async def _mjpeg_stream(
    hub: FrameHub,
    encode: Callable[[np.ndarray], bytes | None],
    authorized: Callable[[], bool] | None = None,
) -> AsyncIterator[bytes]:
    last_seq = -1
    while True:
        if authorized is not None and not authorized():
            return
        frame, seq = hub.latest()
        if frame is not None and seq != last_seq:
            last_seq = seq
            jpeg = await asyncio.to_thread(encode, frame)
            if jpeg is not None and (authorized is None or authorized()):
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(0.1)  # ~10 fps cap for the live view


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Edge Camera Lab", version="0.1.0")
    app.add_middleware(_UploadBodyLimit)

    def _authed(request: Request) -> bool:
        if not state.web_pin:  # auth disabled (dev/tests)
            return True
        token = request.cookies.get(_COOKIE)
        return _valid_token(token, state.session_secret, state.session_epoch, time.time())

    def require_auth(request: Request) -> None:
        if not _authed(request):
            raise HTTPException(status_code=401, detail="giriş gerekli")

    guard = [Depends(require_auth)]

    @app.get("/health")
    async def health() -> dict:
        # Public liveness ONLY — nothing about mode, gallery, or auth posture.
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> str:
        return _INDEX_HTML if _authed(request) else _LOGIN_HTML

    @app.post("/login")
    async def login(body: PinRequest, request: Request, response: Response) -> dict:
        ip = request.client.host if request.client else "?"
        now = time.time()
        recent = [t for t in state.login_failures.get(ip, []) if now - t < _LOGIN_WINDOW]
        if len(recent) >= _LOGIN_MAX_FAILURES:
            state.login_failures[ip] = recent
            raise HTTPException(status_code=429, detail="çok fazla deneme; biraz bekleyin")
        if not state.web_pin or not hmac.compare_digest(body.pin.encode(), state.web_pin.encode()):
            recent.append(now)
            state.login_failures[ip] = recent
            raise HTTPException(status_code=401, detail="hatalı PIN")
        state.login_failures.pop(ip, None)  # success clears the throttle
        response.set_cookie(
            _COOKIE,
            _make_token(state.session_secret, state.session_epoch, now),
            max_age=_SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=state.cookie_secure,
        )
        return {"ok": True}

    @app.post("/logout", dependencies=guard)
    async def logout(response: Response) -> dict:
        state.session_epoch += 1  # revoke every outstanding token, not just this cookie
        response.delete_cookie(_COOKIE)
        return {"ok": True}

    @app.get("/stream", dependencies=guard)
    async def stream(request: Request) -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_stream(state.hub, state.encode_jpeg, lambda: _authed(request)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/events", dependencies=guard)
    async def events(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[dict]:
        return [
            {
                "id": event.id,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.ts)),
                "kind": event.kind,
                "person": event.person_name,
                "detail": event.detail,
                "snapshot": f"/snapshots/{event.snapshot}" if event.snapshot else None,
                "clip": f"/clips/{event.clip}" if event.clip else None,
            }
            for event in state.store.recent_events(limit)
        ]

    @app.get("/snapshots/{relative:path}", dependencies=guard)
    async def snapshot(relative: str) -> FileResponse:
        path = state.store.snapshot_path(relative)
        if path is None:
            raise HTTPException(status_code=404, detail="snapshot not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/clips/{relative:path}", dependencies=guard)
    async def clip(relative: str) -> FileResponse:
        path = state.store.clip_path(relative)
        if path is None:
            raise HTTPException(status_code=404, detail="clip not found")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/mode", dependencies=guard)
    async def get_mode() -> dict:
        return {"mode": state.presence.mode}

    @app.post("/mode", dependencies=guard)
    async def set_mode(request: ModeRequest) -> dict:
        if request.mode not in (MODE_HOME, MODE_AWAY):
            raise HTTPException(status_code=422, detail="mode must be 'home' or 'away'")
        state.presence.set_mode(request.mode)
        return {"mode": state.presence.mode}

    @app.get("/people", dependencies=guard)
    async def people() -> list[dict]:
        return state.store.people()

    @app.delete("/people/{person_id}", dependencies=guard)
    async def delete_person(person_id: int) -> dict:
        if not state.store.remove_person(person_id):
            raise HTTPException(status_code=404, detail="person not found")
        state.matcher_ref.rebuild(state.store.gallery_entries(), state.settings.match_max_distance)
        return {"removed": person_id}

    @app.post("/enroll", dependencies=guard)
    async def enroll(
        name: Annotated[str, Form()],
        photos: Annotated[list[UploadFile], File()],
        is_owner: Annotated[bool, Form()] = False,
    ) -> dict:
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 100:
            raise HTTPException(status_code=422, detail="name must contain 1 to 100 characters")
        if len(photos) > _MAX_UPLOAD_FILES:
            raise HTTPException(status_code=413, detail="too many photos")
        embeddings: list[list[float]] = []
        failed: list[str] = []
        for photo in photos:
            data = await photo.read(_MAX_UPLOAD_FILE_BYTES + 1)
            if len(data) > _MAX_UPLOAD_FILE_BYTES:
                raise HTTPException(status_code=413, detail="photo too large")
            try:
                image = await asyncio.to_thread(state.decode_image, data)
            except (ValueError, RuntimeError):
                image = None
            if (
                image is None
                or image.ndim != 3
                or image.shape[0] * image.shape[1] > _MAX_IMAGE_PIXELS
            ):
                failed.append(photo.filename or "?")
                continue
            faces = await asyncio.to_thread(state.embedder.embed_all, image)
            # Choosing the highest score in a group photo can enroll the wrong
            # person as the owner. Require an unambiguous single face instead.
            if len(faces) != 1:
                failed.append(photo.filename or "?")
                continue
            vector, score = faces[0]
            arr = np.asarray(vector, dtype=np.float64)
            norm = float(np.linalg.norm(arr))
            if (
                arr.shape != (512,)
                or not np.isfinite(arr).all()
                or not np.isfinite(norm)
                or norm <= 0
                or not np.isfinite(score)
            ):
                failed.append(photo.filename or "?")
                continue
            embeddings.append(vector)
        person_id: int | None = None
        if embeddings:
            person_id = state.store.add_person(clean_name, is_owner=is_owner)
            for vector in embeddings:
                state.store.add_embedding(person_id, vector)
            state.matcher_ref.rebuild(
                state.store.gallery_entries(), state.settings.match_max_distance
            )
        return {"person_id": person_id, "enrolled": len(embeddings), "failed": failed}

    return app


_LOGIN_HTML = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edge Camera Lab — Giriş</title>
<style>
  :root{ --bg:#0e0f13; --panel:#171922; --line:#2a2e3d;
         --text:#e7e9ee; --muted:#8b90a0; --accent:#4f8cff; }
  body{ font-family:-apple-system,system-ui,"Segoe UI",Roboto,sans-serif; margin:0;
        color:var(--text);
        background:radial-gradient(900px 500px at 50% -10%, #1b1f30, var(--bg));
        display:grid; place-items:center; min-height:100vh; }
  form{ background:var(--panel); border:1px solid var(--line); padding:30px 26px; border-radius:16px;
        width:280px; text-align:center; box-shadow:0 20px 60px rgba(0,0,0,.45); }
  h1{ font-size:22px; margin:0 0 16px; }
  input{ width:100%; box-sizing:border-box; padding:12px; border-radius:10px; border:1px solid var(--line);
         background:var(--bg); color:var(--text); font-size:17px; text-align:center; }
  button{ margin-top:12px; width:100%; border:0; border-radius:10px; padding:12px; font-size:15px;
          background:var(--accent); color:#06122e; font-weight:800; cursor:pointer; }
  .hint{ color:var(--muted); font-size:12px; margin-top:8px; }
  #err{ color:#ff8a8a; font-size:13px; min-height:18px; margin-top:8px; }
</style>
</head>
<body>
<form id="f">
  <h1>🏠 Edge Camera Lab</h1>
  <input id="pin" type="password" placeholder="Şifre"
         autofocus autocomplete="current-password">
  <div class="hint">Kelime ya da rakam — kayıtlı parolanı gir.</div>
  <button type="submit">Giriş</button>
  <div id="err"></div>
</form>
<script>
document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = document.getElementById("err");
  err.textContent = "…";
  const r = await fetch("/login", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pin: document.getElementById("pin").value }) });
  if (r.ok) { location.reload(); } else { err.textContent = "Hatalı PIN"; }
});
</script>
</body>
</html>
"""


_INDEX_HTML = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edge Camera Lab</title>
<meta name="theme-color" content="#0e0f13">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Edge Camera Lab">
<link rel="manifest" href="data:application/json,%7B%22name%22%3A%22Edge Camera Lab%22%2C%22short_name%22%3A%22Edge Camera Lab%22%2C%22display%22%3A%22standalone%22%2C%22background_color%22%3A%22%230e0f13%22%2C%22theme_color%22%3A%22%230e0f13%22%2C%22start_url%22%3A%22%2F%22%7D">
<style>
  :root{ --bg:#0e0f13; --panel:#171922; --panel2:#20232f; --line:#2a2e3d; --text:#e7e9ee;
         --muted:#8b90a0; --accent:#4f8cff; --home:#2ecc71; --away:#ff4d4f; --r:14px; }
  *{ box-sizing:border-box; }
  body{ font-family:-apple-system,system-ui,"Segoe UI",Roboto,sans-serif; margin:0; color:var(--text);
        background:radial-gradient(1100px 560px at 72% -12%, #1b1f30, var(--bg)); min-height:100vh; }
  header{ position:sticky; top:0; z-index:10; display:flex; align-items:center; gap:12px;
          padding:12px 18px; background:rgba(14,15,19,.82); backdrop-filter:blur(10px);
          border-bottom:1px solid var(--line); }
  header h1{ font-size:17px; margin:0; letter-spacing:.3px; }
  .grow{ flex:1; }
  #modeBadge{ padding:5px 12px; border-radius:999px; font-size:12px; font-weight:800;
              letter-spacing:.6px; color:#06210f; }
  .seg{ display:inline-flex; background:var(--panel2); border:1px solid var(--line);
        border-radius:999px; padding:3px; }
  .seg button{ border:0; background:transparent; color:var(--muted); font-weight:700; font-size:13px;
               padding:7px 15px; border-radius:999px; cursor:pointer; transition:.15s; }
  .seg button.on.home{ background:var(--home); color:#06210f; }
  .seg button.on.away{ background:var(--away); color:#2a0708; }
  main{ display:grid; grid-template-columns:1.7fr 1fr; gap:16px; padding:16px;
        max-width:1180px; margin:0 auto; }
  @media (max-width:860px){ main{ grid-template-columns:1fr; padding:12px; } }
  .col{ display:flex; flex-direction:column; gap:16px; min-width:0; }
  section{ background:var(--panel); border:1px solid var(--line); border-radius:var(--r); padding:16px; }
  section h2{ font-size:12px; text-transform:uppercase; letter-spacing:.9px; color:var(--muted); margin:0 0 12px; }
  .live-wrap{ position:relative; border-radius:12px; overflow:hidden; background:#000; aspect-ratio:16/9; }
  img#live{ width:100%; height:100%; object-fit:cover; display:block; }
  .live-tag{ position:absolute; top:10px; left:10px; display:flex; align-items:center; gap:6px;
             background:rgba(255,77,79,.92); color:#fff; font-size:11px; font-weight:800;
             padding:4px 9px; border-radius:999px; }
  .live-tag i{ width:7px; height:7px; border-radius:50%; background:#fff; animation:blink 1.4s infinite; }
  @keyframes blink{ 50%{ opacity:.25; } }
  .events{ display:flex; flex-direction:column; gap:8px; max-height:62vh; overflow:auto; }
  .ev{ display:flex; align-items:center; gap:12px; padding:8px; border-radius:12px;
       background:var(--panel2); border:1px solid transparent; }
  .ev.alarm{ border-color:rgba(255,77,79,.45); background:rgba(255,77,79,.09); }
  .ev .thumb{ width:54px; height:54px; border-radius:9px; object-fit:cover; background:#000;
              flex:none; border:1px solid var(--line); }
  .ev .thumb.ph{ display:flex; align-items:center; justify-content:center; font-size:22px; }
  .ev .meta{ min-width:0; flex:1; }
  .ev .title{ font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .ev .sub{ color:var(--muted); font-size:12px; margin-top:2px; }
  .ev a.clip{ color:var(--accent); text-decoration:none; font-size:19px; }
  a{ color:var(--accent); }
  input[type=text], input[type=file]{ width:100%; padding:9px 10px; border-radius:9px;
      border:1px solid var(--line); background:var(--bg); color:var(--text); font-size:14px; }
  label.chk{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:14px; }
  form p{ margin:0 0 10px; }
  button.act{ border:0; border-radius:10px; padding:9px 15px; cursor:pointer; font-weight:700;
              font-size:13px; background:var(--accent); color:#06122e; }
  button.ghost{ background:var(--panel2); color:var(--text); border:1px solid var(--line); }
  .people{ display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }
  .chip{ display:flex; align-items:center; gap:8px; background:var(--panel2); border:1px solid var(--line);
         border-radius:999px; padding:6px 12px; font-size:13px; }
  .chip .x{ color:var(--muted); text-decoration:none; }
  .muted{ color:var(--muted); font-size:13px; }
  .sec-head{ display:flex; align-items:center; gap:8px; margin:0 0 12px; }
  .sec-head h2{ margin:0; }
  .filter{ margin-left:auto; background:var(--panel2); color:var(--text); border:1px solid var(--line);
           border-radius:8px; padding:5px 8px; font-size:12px; }
  .toast{ position:fixed; left:50%; bottom:20px; transform:translate(-50%,40px); background:var(--away);
          color:#fff; font-weight:800; padding:12px 18px; border-radius:12px; z-index:50;
          box-shadow:0 12px 40px rgba(0,0,0,.5); opacity:0; pointer-events:none; transition:.25s; }
  .toast.show{ opacity:1; transform:translate(-50%,0); }
</style>
</head>
<body>
<header>
  <h1>🏠 Edge Camera Lab</h1>
  <span id="modeBadge">…</span>
  <span class="grow"></span>
  <div class="seg">
    <button id="mHome" class="home" onclick="setMode('home')">Evde</button>
    <button id="mAway" class="away" onclick="setMode('away')">Dışarıda</button>
  </div>
</header>
<main>
  <div class="col">
    <section>
      <h2>Canlı görüntü</h2>
      <div class="live-wrap">
        <img id="live" src="/stream" alt="canlı yayın">
        <div class="live-tag"><i></i>CANLI</div>
      </div>
    </section>
    <section>
      <div class="sec-head"><h2>Olaylar</h2>
        <select id="filter" class="filter">
          <option value="all">Tümü</option>
          <option value="known_person">✅ Tanınan</option>
          <option value="unknown_person">🚨 Tanınmayan</option>
          <option value="person_detected">🚷 İnsan</option>
          <option value="cat_activity">🐱 Kedi</option>
        </select>
      </div>
      <div class="events" id="events"></div>
    </section>
  </div>
  <div class="col">
    <section>
      <h2>Kişi ekle</h2>
      <form id="enrollForm">
        <p><input type="text" name="name" placeholder="İsim (örn. Demo Person)" required></p>
        <p><label class="chk"><input type="checkbox" name="is_owner" value="true"> Ev sahibi</label></p>
        <p><input type="file" name="photos" accept="image/*" multiple required></p>
        <p><button class="act ghost" type="submit">Fotoğraflardan kaydet</button>
           <span id="enrollResult" class="muted"></span></p>
      </form>
      <div id="people" class="people"></div>
    </section>
  </div>
</main>
<div id="toast" class="toast"></div>
<script>
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const KINDS = {
  known_person: (e) => "✅ " + (e.person ? esc(e.person) : "Tanınan kişi"),
  unknown_person: () => "🚨 Tanınmayan kişi",
  person_detected: () => "🚷 İnsan algılandı (yüz yok)",
  cat_activity: () => "🐱 Kedi hareketi",
  mode_change: (e) => "🔁 Mod: " + (e.detail === "away" ? "dışarıda" : "evde"),
};
const ALARM = new Set(["unknown_person", "person_detected"]);
let allEvents = [], lastSeen = {}, catSeen = null, seenMaxId = -1, firstLoad = true;
function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = "square"; o.frequency.value = 880; o.connect(g); g.connect(ctx.destination);
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.15, ctx.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.4);
    o.start(); o.stop(ctx.currentTime + 0.4);
  } catch (err) {}
}
let toastT;
function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 6000);
}
async function refreshMode() {
  const d = await (await fetch("/mode")).json();
  const away = d.mode === "away";
  const b = document.getElementById("modeBadge");
  b.textContent = away ? "DIŞARIDA" : "EVDE";
  b.style.background = away ? "var(--away)" : "var(--home)";
  b.style.color = away ? "#2a0708" : "#06210f";
  document.getElementById("mHome").classList.toggle("on", !away);
  document.getElementById("mAway").classList.toggle("on", away);
}
async function setMode(mode) {
  await fetch("/mode", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }) });
  refreshMode(); refreshEvents();
}
function renderEvents() {
  const f = document.getElementById("filter").value;
  const rows = f === "all" ? allEvents : allEvents.filter(e => e.kind === f);
  const el = document.getElementById("events");
  if (!rows.length) { el.innerHTML = '<div class="muted">Olay yok.</div>'; return; }
  el.innerHTML = rows.map(e => {
    const label = (KINDS[e.kind] || ((x) => esc(x.kind)))(e);
    const thumb = e.snapshot
      ? `<a href="${e.snapshot}" target="_blank"><img class="thumb" loading="lazy" src="${e.snapshot}" alt=""></a>`
      : `<div class="thumb ph">${e.kind === "cat_activity" ? "🐱" : "📷"}</div>`;
    const clip = e.clip ? `<a class="clip" href="${e.clip}" target="_blank" title="video">🎥</a>` : "";
    const sub = e.time + (e.detail ? " · " + esc(e.detail) : "");
    return `<div class="ev ${ALARM.has(e.kind) ? "alarm" : ""}">${thumb}` +
      `<div class="meta"><div class="title">${label}</div><div class="sub">${sub}</div></div>${clip}</div>`;
  }).join("");
}
async function refreshEvents() {
  const rows = await (await fetch("/events?limit=60")).json();
  allEvents = rows;
  lastSeen = {}; catSeen = null;
  for (const e of rows) {
    if (e.kind === "known_person" && e.person && !(e.person in lastSeen)) lastSeen[e.person] = e.time;
    if (e.kind === "cat_activity" && catSeen === null) catSeen = e.time;
  }
  const newest = rows.length ? Math.max(...rows.map(e => e.id)) : -1;
  if (!firstLoad) {
    const fresh = rows.filter(e => e.id > seenMaxId && ALARM.has(e.kind));
    if (fresh.length) {
      toast(fresh[0].kind === "unknown_person" ? "🚨 Tanınmayan kişi algılandı!" : "🚷 İnsan algılandı (yüz yok)!");
      beep();
    }
  }
  seenMaxId = Math.max(seenMaxId, newest); firstLoad = false;
  renderEvents(); refreshPeople();
}
async function removePerson(id, name) {
  if (!confirm(`${name} silinsin mi? (tüm yüz kayıtlarıyla birlikte)`)) return;
  await fetch(`/people/${id}`, { method: "DELETE" });
  refreshPeople();
}
async function refreshPeople() {
  const rows = await (await fetch("/people")).json();
  const el = document.getElementById("people");
  const hhmm = (t) => esc(String(t).slice(11, 16));
  let html = rows.length
    ? rows.map(p => {
        const seen = lastSeen[p.name] ? ` · <span class="muted">son ${hhmm(lastSeen[p.name])}</span>` : "";
        return `<span class="chip">${p.is_owner ? "👑" : "👤"} ${esc(p.name)} ` +
          `<span class="muted">${p.embeddings} foto</span>${seen} ` +
          `<button class="x" type="button" title="sil" data-person-id="${p.id}">✕</button></span>`;
      }).join("")
    : '<span class="muted">Kayıtlı kişi yok — önce kendini ekle.</span>';
  if (catSeen) html += `<span class="chip">🐱 Kedi <span class="muted">son ${hhmm(catSeen)}</span></span>`;
  el.innerHTML = html;
  const names = new Map(rows.map(p => [String(p.id), p.name]));
  for (const button of el.querySelectorAll("button[data-person-id]")) {
    button.addEventListener("click", () =>
      removePerson(Number(button.dataset.personId), names.get(button.dataset.personId)));
  }
}
document.getElementById("filter").addEventListener("change", renderEvents);
document.getElementById("enrollForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = document.getElementById("enrollResult");
  out.textContent = "kaydediliyor…";
  const resp = await fetch("/enroll", { method: "POST", body: new FormData(ev.target) });
  const d = await resp.json();
  const failed = d.failed && d.failed.length ? ", hatalı: " + d.failed.join(", ") : "";
  out.textContent = resp.ok ? `✔ ${d.enrolled} fotoğraf işlendi${failed}` : "hata";
  refreshPeople();
});
refreshMode(); refreshEvents();
setInterval(() => { refreshMode(); refreshEvents(); }, 5000);
</script>
</body>
</html>
"""
