import asyncio
import json
import os
import re
import time
import uuid
import threading
import webbrowser
import subprocess
import shutil
import random
from contextlib import asynccontextmanager
from typing import Dict, List, Set, Optional, Any, Tuple

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ============================================================
# Discord AI2026 (single-folder)
# Put main.py + index.html in the SAME folder
# Run:  python main.py
#
# Requirements:
#   pip install fastapi uvicorn[standard] httpx pydantic
#
# Defaults (you can override via env vars):
#   CHAT_MODEL=llama3.2:3b
#   GATE_MODEL=llama3.2:1b
#
# Dev Mode:
#   UI toggle sends {"type":"set_debug","enabled":true} to WS.
#   Server will then stream debug events to that client only.
#
# This version keeps all core behavior, and adds OPTIONAL "Discord-like"
# features (UI toggles control usage):
#  - Message replies (reply_to)
#  - Message edits
#  - Emoji reactions
#  - Pin messages + pins list
#  - Simple search endpoint
#  - Presence (online/idle/typing/offline) computed server-side
#  - Optional human-like pacing per character (disabled by default)
#  - Optional split-into-multiple messages per character (disabled by default)
# ============================================================

APP_HOST = "127.0.0.1"
APP_PORT = 8000

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "index.html")
STATE_PATH = os.path.join(HERE, "state.json")

CHAT_MODEL_DEFAULT = os.getenv("CHAT_MODEL", "llama3.2:3b")
GATE_MODEL_DEFAULT = os.getenv("GATE_MODEL", "llama3.2:1b")
IDLE_NUDGE_SECONDS = int(os.getenv("IDLE_NUDGE_SECONDS", "300"))

MAX_AUTO_MESSAGES_PER_MINUTE = int(os.getenv("MAX_AUTO_MESSAGES_PER_MINUTE", "12"))
MIN_SECONDS_BETWEEN_CHARACTER_MESSAGES = float(os.getenv("MIN_SECONDS_BETWEEN_CHARACTER_MESSAGES", "2.0"))

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_VERSION_ENDPOINT = f"{OLLAMA_BASE_URL}/api/version"
OLLAMA_START_MODE = os.getenv("OLLAMA_START_MODE", "on_demand").strip().lower()  # always|on_demand|never
# OpenRouter (optional cloud backend)
LLM_PROVIDER_DEFAULT = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()  # "ollama" | "openrouter" | "puter"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_CHAT_ENDPOINT = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "mistralai/devstral-2512:free").strip()
# Puter (optional cloud backend)
PUTER_BASE_URL = os.getenv("PUTER_BASE_URL", "https://api.puter.com/v1").rstrip("/")
PUTER_CHAT_ENDPOINT = f"{PUTER_BASE_URL}/chat/completions"
PUTER_API_KEY = os.getenv("PUTER_API_KEY", "").strip()
PUTER_DEFAULT_MODEL = os.getenv("PUTER_DEFAULT_MODEL", "").strip()
# Optional metadata headers recommended by OpenRouter
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()   # e.g. "https://example.com"
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Discord AI2026").strip()
OPENROUTER_PRESENCE_PENALTY = float(os.getenv("OPENROUTER_PRESENCE_PENALTY", "0.6"))
OPENROUTER_FREQUENCY_PENALTY = float(os.getenv("OPENROUTER_FREQUENCY_PENALTY", "0.3"))


def _norm_provider(p: str) -> str:
    p = (p or "").strip().lower()
    if p in ("openrouter", "or", "router", "open-router"):
        return "openrouter"
    if p in ("puter", "puterjs", "puter-js", "puter js"):
        return "puter"
    return "ollama"


def _provider_label(p: str) -> str:
    p = _norm_provider(p)
    if p == "openrouter":
        return "OpenRouter"
    if p == "puter":
        return "Puter"
    return "Ollama"


async def openrouter_chat(
    model: str,
    messages: List[dict],
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    key = (api_key or OPENROUTER_API_KEY or "").strip()
    if not key:
        raise RuntimeError("OpenRouter API key missing. Set OPENROUTER_API_KEY env var.")

    payload = {
        "model": model or OPENROUTER_DEFAULT_MODEL,
        "stream": False,
        "temperature": float(temperature),
        "messages": messages,
    }
    if OPENROUTER_PRESENCE_PENALTY != 0.0:
        payload["presence_penalty"] = float(OPENROUTER_PRESENCE_PENALTY)
    if OPENROUTER_FREQUENCY_PENALTY != 0.0:
        payload["frequency_penalty"] = float(OPENROUTER_FREQUENCY_PENALTY)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # OpenRouter recommends these headers for attribution (optional)
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    timeout = max(5.0, float(timeout))
    last_err: Optional[Exception] = None

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(OPENROUTER_CHAT_ENDPOINT, json=payload, headers=headers)
                r.raise_for_status()
                try:
                    data = r.json()
                except Exception as e:
                    raise RuntimeError("OpenRouter returned invalid JSON") from e

            choice0 = (data.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            content = msg.get("content")
            if content is None and "delta" in choice0:
                content = (choice0.get("delta") or {}).get("content", "")
            return (content or "").strip()
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "").strip()
            detail = body[:300] if body else str(e)
            raise RuntimeError(f"OpenRouter HTTP {e.response.status_code}: {detail}") from e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(0.4 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter request failed: {e}") from e

    if last_err:
        raise RuntimeError(f"OpenRouter request failed: {last_err}")
    return ""


async def puter_chat(
    model: str,
    messages: List[dict],
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    if not PUTER_BASE_URL:
        raise RuntimeError("Puter base URL missing. Set PUTER_BASE_URL.")

    payload: Dict[str, Any] = {
        "stream": False,
        "temperature": float(temperature),
        "messages": messages,
    }
    if (model or "").strip():
        payload["model"] = model

    headers = {"Content-Type": "application/json"}
    key = (api_key or PUTER_API_KEY or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    timeout = max(5.0, float(timeout))
    last_err: Optional[Exception] = None

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(PUTER_CHAT_ENDPOINT, json=payload, headers=headers)
                r.raise_for_status()
                try:
                    data = r.json()
                except Exception as e:
                    raise RuntimeError("Puter returned invalid JSON") from e

            choice0 = (data.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            content = msg.get("content")
            if content is None and "delta" in choice0:
                content = (choice0.get("delta") or {}).get("content", "")
            return (content or "").strip()
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "").strip()
            detail = body[:300] if body else str(e)
            raise RuntimeError(f"Puter HTTP {e.response.status_code}: {detail}") from e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(0.4 * (attempt + 1))
                continue
            raise RuntimeError(f"Puter request failed: {e}") from e

    if last_err:
        raise RuntimeError(f"Puter request failed: {last_err}")
    return ""

async def llm_chat(provider: str, model: str, messages: List[dict], temperature: float = 0.7) -> str:
    provider = _norm_provider(provider)
    if provider == "openrouter":
        return await openrouter_chat(model=model, messages=messages, temperature=temperature)
    if provider == "puter":
        return await puter_chat(model=model, messages=messages, temperature=temperature)
    return await ollama_chat(model=model, messages=messages, temperature=temperature)


def openrouter_configured() -> bool:
    return bool(OPENROUTER_API_KEY)


def puter_configured() -> bool:
    return bool(PUTER_BASE_URL)




# Presence tuning (purely cosmetic)
PRESENCE_TICK_SECONDS = 5
DEFAULT_IDLE_AFTER_SECONDS = 180


# -------------------------
# Pydantic models
# -------------------------
class CharacterCreate(BaseModel):
    name: str

    # LLM backend selection
    provider: str = Field(default=LLM_PROVIDER_DEFAULT, description="ollama|openrouter|puter")
    model: str = CHAT_MODEL_DEFAULT

    system_prompt: str = "You are a helpful AI."

    # Gate (who decides if the character should speak)
    gate_provider: str = Field(default=LLM_PROVIDER_DEFAULT, description="ollama|openrouter|puter")
    gate_model: str = GATE_MODEL_DEFAULT
    gate_interval_seconds: int = 30

    enabled: bool = True

    # Optional (Discord-like flavor)
    about: str = ""
    profile_color: str = ""  # CSS color (e.g. "#5865F2") or empty for auto
    idle_after_seconds: int = DEFAULT_IDLE_AFTER_SECONDS

    # Optional "human" behavior (OFF by default to preserve old behavior)
    humanize_pacing: bool = False


class CharacterUpdate(BaseModel):
    name: Optional[str] = None

    provider: Optional[str] = None
    model: Optional[str] = None

    system_prompt: Optional[str] = None

    gate_provider: Optional[str] = None
    gate_model: Optional[str] = None
    gate_interval_seconds: Optional[int] = None

    enabled: Optional[bool] = None
    about: Optional[str] = None
    profile_color: Optional[str] = None
    idle_after_seconds: Optional[int] = None
    humanize_pacing: Optional[bool] = None


class Character(BaseModel):
    id: str
    name: str

    provider: str
    model: str

    system_prompt: str

    gate_provider: str
    gate_model: str
    gate_interval_seconds: int

    enabled: bool

    about: str = ""
    profile_color: str = ""
    idle_after_seconds: int = DEFAULT_IDLE_AFTER_SECONDS
    humanize_pacing: bool = False


class ConversationCreate(BaseModel):
    title: str = "general"
    character_ids: List[str] = Field(default_factory=list)


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    character_ids: Optional[List[str]] = None


class Conversation(BaseModel):
    id: str
    title: str
    character_ids: List[str]


class MessageCreate(BaseModel):
    author_type: str  # "user" or "character"
    author_id: str    # "user" or character_id
    content: str
    reply_to: Optional[str] = None


class MessageEdit(BaseModel):
    editor_type: str  # "user" or "character"
    editor_id: str    # "user" or character_id
    content: str


class ReactionToggle(BaseModel):
    emoji: str
    actor_type: str  # "user" or "character"
    actor_id: str    # "user" or character_id


class PinToggle(BaseModel):
    actor_type: str  # "user" or "character"
    actor_id: str    # "user" or character_id
    pinned: Optional[bool] = None  # if omitted, toggles


class Message(BaseModel):
    id: str
    conversation_id: str
    author_type: str
    author_id: str
    content: str
    ts: float

    # Optional Discord-like features
    edited_ts: Optional[float] = None
    reactions: Dict[str, List[str]] = Field(default_factory=dict)  # emoji -> [actor_id,...]
    pinned: bool = False
    reply_to: Optional[str] = None


class UserProfile(BaseModel):
    username: str = "User"


class PauseToggle(BaseModel):
    enabled: bool = False


# -------------------------
# In-memory state
# -------------------------
CHARACTERS: Dict[str, Character] = {}
CONVERSATIONS: Dict[str, Conversation] = {}
MESSAGES: Dict[str, List[Message]] = {}

LAST_GATE_CHECK: Dict[str, Dict[str, float]] = {}  # conversation_id -> char_id -> ts
CHAR_BUSY: Dict[str, bool] = {}                    # char_id -> busy?
LAST_RESPONDED_TO: Dict[str, Dict[str, float]] = {}  # conversation_id -> char_id -> last non-self msg ts handled
LAST_IDLE_NUDGE: Dict[str, Dict[str, float]] = {}    # conversation_id -> char_id -> last idle nudge msg ts handled

# Cosmetic "last activity" for presence
CHAR_LAST_ACTIVE: Dict[str, float] = {}            # char_id -> ts (message committed)
USER_PROFILE: Dict[str, str] = {"username": "User"}
PAUSE_ALL_CALLS = False
STATE_LOCK = threading.Lock()


def _serialize_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "ts": time.time(),
        "user_profile": USER_PROFILE,
        "settings": {
            "ollama_start_mode": OLLAMA_START_MODE,
        },
        "characters": [c.model_dump() for c in CHARACTERS.values()],
        "conversations": [c.model_dump() for c in CONVERSATIONS.values()],
        "messages": {
            cid: [m.model_dump() for m in msgs]
            for cid, msgs in MESSAGES.items()
        },
    }


def _save_state():
    try:
        with STATE_LOCK:
            payload = _serialize_state()
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save state: {e}")


def _apply_state(data: Dict[str, Any]) -> bool:
    try:
        chars = data.get("characters") or []
        convs = data.get("conversations") or []
        msgs = data.get("messages") or {}
        profile = data.get("user_profile") or {}
        settings = data.get("settings") or {}
        if not chars and not convs and not msgs:
            return False

        CHARACTERS.clear()
        CONVERSATIONS.clear()
        MESSAGES.clear()
        CHAR_BUSY.clear()
        CHAR_LAST_ACTIVE.clear()
        LAST_GATE_CHECK.clear()
        LAST_RESPONDED_TO.clear()
        LAST_IDLE_NUDGE.clear()

        USER_PROFILE["username"] = _normalize_username(profile.get("username", "User"))
        _set_ollama_start_mode(settings.get("ollama_start_mode") or OLLAMA_START_MODE)

        for c in chars:
            char = Character(**c)
            CHARACTERS[char.id] = char
            CHAR_BUSY[char.id] = False

        for c in convs:
            conv = Conversation(**c)
            CONVERSATIONS[conv.id] = conv
            MESSAGES.setdefault(conv.id, [])
            LAST_GATE_CHECK.setdefault(conv.id, {})
            LAST_RESPONDED_TO.setdefault(conv.id, {})
            LAST_IDLE_NUDGE.setdefault(conv.id, {})

        for cid, items in msgs.items():
            out = []
            for m in items or []:
                out.append(Message(**m))
            MESSAGES[cid] = out

        # Set last active based on last character message per char
        for cid, items in MESSAGES.items():
            for m in items:
                if m.author_type == "character":
                    CHAR_LAST_ACTIVE[m.author_id] = max(
                        CHAR_LAST_ACTIVE.get(m.author_id, 0.0),
                        float(m.ts or 0.0),
                    )
        return True
    except Exception as e:
        print(f"[WARN] Failed to load state: {e}")
        return False


def _load_state():
    if not os.path.exists(STATE_PATH):
        return False
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _apply_state(data or {})
    except Exception as e:
        print(f"[WARN] Failed to read state: {e}")
        return False


# -------------------------
# WebSocket manager (supports per-client debug toggle)
# -------------------------
class WSManager:
    def __init__(self):
        self.rooms: Dict[str, Set[WebSocket]] = {}
        self.meta: Dict[WebSocket, Dict[str, Any]] = {}

    async def join(self, conversation_id: str, ws: WebSocket):
        await ws.accept()
        self.rooms.setdefault(conversation_id, set()).add(ws)
        self.meta[ws] = {"debug": False}

    def leave(self, conversation_id: str, ws: WebSocket):
        if conversation_id in self.rooms:
            self.rooms[conversation_id].discard(ws)
            if not self.rooms[conversation_id]:
                del self.rooms[conversation_id]
        self.meta.pop(ws, None)

    def set_debug(self, ws: WebSocket, enabled: bool):
        if ws in self.meta:
            self.meta[ws]["debug"] = bool(enabled)

    async def broadcast(self, conversation_id: str, payload: dict):
        """Send to everyone in the room."""
        if conversation_id not in self.rooms:
            return
        dead = []
        for ws in list(self.rooms[conversation_id]):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.leave(conversation_id, ws)

    async def broadcast_debug(self, conversation_id: str, payload: dict):
        """Send only to clients with debug enabled."""
        if conversation_id not in self.rooms:
            return
        dead = []
        for ws in list(self.rooms[conversation_id]):
            try:
                if self.meta.get(ws, {}).get("debug", False):
                    await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.leave(conversation_id, ws)


ws_manager = WSManager()


# -------------------------
# Debug event helper
# -------------------------
async def debug_event(conversation_id: str, level: str, stage: str, data: dict):
    payload = {
        "type": "debug",
        "data": {
            "ts": time.time(),
            "level": level,
            "stage": stage,
            "data": data,
        }
    }
    await ws_manager.broadcast_debug(conversation_id, payload)


def _truncate(s: str, n: int = 600) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f" …(+{len(s)-n} chars)"


# -------------------------
# Ollama helpers (best-effort auto-start)
# -------------------------
def _find_ollama_exe() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe

    local_appdata = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "")
    candidates = [
        os.path.join(local_appdata, "Programs", "Ollama", "ollama.exe"),
        os.path.join(program_files, "Ollama", "ollama.exe"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _set_ollama_start_mode(mode: str):
    global OLLAMA_START_MODE
    mode = (mode or "").strip().lower()
    if mode not in ("always", "on_demand", "never"):
        mode = "on_demand"
    OLLAMA_START_MODE = mode


def _maybe_start_ollama():
    if OLLAMA_START_MODE == "never":
        return
    if OLLAMA_START_MODE in ("always", "on_demand"):
        if not ollama_is_up():
            ensure_ollama_running()


def ollama_is_up(timeout_s: float = 0.7) -> bool:
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(OLLAMA_VERSION_ENDPOINT)
            return r.status_code == 200
    except Exception:
        return False


def ensure_ollama_running():
    if ollama_is_up():
        print("[OK] Ollama is running.")
        return

    exe = _find_ollama_exe()
    if not exe:
        print("[WARN] Ollama not found on PATH/common install paths.")
        print("       Start Ollama manually (GUI is fine).")
        return

    print("[INFO] Starting Ollama with `ollama serve` (best-effort)...")
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags
        )
    except Exception as e:
        print(f"[WARN] Failed to start Ollama automatically: {e}")
        return

    for _ in range(40):
        if ollama_is_up():
            print("[OK] Ollama is up.")
            return
        time.sleep(0.2)

    print("[WARN] Ollama did not become ready. Start it manually if needed.")


async def ollama_chat(model: str, messages: List[dict], temperature: float = 0.7) -> str:
    _maybe_start_ollama()
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(OLLAMA_CHAT_ENDPOINT, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]


async def ollama_tags() -> List[str]:
    _maybe_start_ollama()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(OLLAMA_TAGS_ENDPOINT)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name")
            if name:
                models.append(name)
        return models


# -------------------------
# Conversation participant names
# -------------------------
def _normalize_username(name: Optional[str]) -> str:
    clean = (name or "").strip()
    if not clean:
        return "User"
    return clean[:48]


def _resolve_user_display(author_id: Optional[str]) -> str:
    author = (author_id or "").strip()
    if not author or author.lower() == "user":
        return _normalize_username(USER_PROFILE.get("username", "User"))
    return _normalize_username(author)


def get_participant_names(conversation_id: str) -> List[str]:
    conv = CONVERSATIONS.get(conversation_id)
    if not conv:
        return []
    names = []
    for cid in conv.character_ids:
        c = CHARACTERS.get(cid)
        if c:
            names.append(c.name)
    return names


# -------------------------
# Context building
# -------------------------
def build_context_for_character(conversation_id: str, character: Character, max_messages: int = 30) -> List[dict]:
    """
    Chat context includes:
    - The character's system prompt
    - A list of all participant names in the conversation
    - Recent messages

    IMPORTANT:
    - Gate outputs are never saved into MESSAGES, so they can't contaminate context.
    - Extra safeguard: ignore ultra-short gate-like leaked lines if they ever appear.
    """
    msgs = MESSAGES.get(conversation_id, [])[-max_messages:]
    all_msgs = MESSAGES.get(conversation_id, [])

    participants = get_participant_names(conversation_id)
    participants_line = ", ".join(participants) if participants else "(none)"

    idle_s = max(0, int(time.time() - _latest_message_ts(conversation_id)))
    last_msg = all_msgs[-1] if all_msgs else None
    followup_hint = ""
    if last_msg and last_msg.author_type == "character" and last_msg.author_id == character.id:
        followup_hint = (
            "- You already spoke in the most recent message.\n"
            "- Do NOT repeat any sentence from your last message.\n"
            "- Continue with new information or a new question only.\n"
        )
    # allow consecutive replies; no "last message" restriction
    transcript_lines = []
    for m in all_msgs:
        if m.author_type == "user":
            name = _resolve_user_display(m.author_id)
        else:
            name = CHARACTERS.get(m.author_id).name if m.author_id in CHARACTERS else "Character"
        transcript_lines.append(f"{name}: {m.content}")
    transcript_block = "\n".join(transcript_lines) if transcript_lines else "(no messages)"

    system = (
        f"{character.system_prompt}\n\n"
        f"You are '{character.name}' in a group chat.\n"
        f"Participants (AI): {participants_line}\n"
        f"Time since last message: {idle_s} seconds.\n"
        f"Rules:\n"
        f"- Speak only as {character.name}.\n"
        f"- Do not output meta/debug/gating decisions.\n"
        f"- Never repeat a previous message verbatim. Speak like a Discord user, and be concise. Speak like a real human\n"
        f"{followup_hint}"
        f"make your responses short and concise and limit emoji use. Do not use italics and do not narrate your actions. Type in a style someone on a live chat would use"
        f"transcription of the chat:\n{transcript_block}\n"
    )

    ctx = [{"role": "system", "content": system}]
    return ctx


def _latest_nonself_message_ts(conversation_id: str, character_id: str) -> float:
    msgs = MESSAGES.get(conversation_id, [])
    for m in reversed(msgs):
        if not (m.author_type == "character" and m.author_id == character_id):
            return float(m.ts)
    return 0.0


def _latest_message_ts(conversation_id: str) -> float:
    msgs = MESSAGES.get(conversation_id, [])
    if not msgs:
        return 0.0
    return float(msgs[-1].ts)

def _auto_rate_limited(conversation_id: str, window_s: int = 60) -> bool:
    """Prevent runaway AI-to-AI loops.

    Returns True if the number of CHARACTER messages in the last window_s seconds
    meets/exceeds MAX_AUTO_MESSAGES_PER_MINUTE.
    """
    limit = max(1, int(MAX_AUTO_MESSAGES_PER_MINUTE))
    if limit <= 0:
        return False
    now = time.time()
    msgs = MESSAGES.get(conversation_id, [])
    if not msgs:
        return False
    cutoff = now - float(window_s)
    cnt = 0
    for m in reversed(msgs):
        if float(m.ts) < cutoff:
            break
        if m.author_type == 'character':
            cnt += 1
            if cnt >= limit:
                return True
    return False

def _last_message_by_character(conversation_id: str, character_id: str) -> Optional[str]:
    msgs = MESSAGES.get(conversation_id, [])
    for m in reversed(msgs):
        if m.author_type == "character" and m.author_id == character_id:
            return m.content or ""
    return None

def _norm_text(s: Optional[str]) -> str:
    return " ".join((s or "").split()).strip().lower()


# -------------------------
# Gate logic (YES/NO by count — tiny-model forgiving)
# -------------------------
YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

async def judge_should_respond(conversation_id: str, character: Character) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "respond": bool,
        "raw": str,
        "decision": str,
        "prompt_preview": str,
        "yes_count": int,
        "no_count": int
      }

    Decision rule:
      - respond True if yes_count > no_count
      - respond False if no_count >= yes_count
      - if both zero or tie -> False
    """
    msgs = MESSAGES.get(conversation_id, [])[-20:]
    if not msgs:
        return {
            "respond": False, "raw": "", "decision": "NO_MESSAGES",
            "prompt_preview": "", "yes_count": 0, "no_count": 0
        }

    transcript_lines = []
    for m in msgs:
        if m.author_type == "user":
            user_name = _resolve_user_display(m.author_id)
            transcript_lines.append(f"{user_name}: {m.content}")
        else:
            name = CHARACTERS.get(m.author_id).name if m.author_id in CHARACTERS else "Character"
            transcript_lines.append(f"{name}: {m.content}")

    participants = get_participant_names(conversation_id)
    participants_line = ", ".join(participants) if participants else "(none)"

    # IMPORTANT: The gate output is NOT stored anywhere; it's only used for this decision + dev debug.
    gate_prompt = f"""

Character considering speaking: "{character.name}"
Other participants (AI): {participants_line}
"{character.name}" is currently in a live chatting app
Time since last message: {max(0, int(time.time() - _latest_message_ts(conversation_id)))} seconds.

You are deciding if "{character.name}" should reply RIGHT NOW by saying YES or NO at the end of your response. add a reason why you choose your answer at the beginning of your response.

keep your reason short and consice. you MUST use YES or NO in your response.
If chat has been idle for {IDLE_NUDGE_SECONDS} seconds or more, you CAN reply to keep the chat going. 
You get to decide when you want to speak You don't have to follow any orders but you can.

Transcript of the live chat:
{chr(10).join(transcript_lines)}

""".strip()

    prompt_preview = _truncate(gate_prompt, 700)

    try:
        raw = await llm_chat(
            provider=character.gate_provider,
            model=character.gate_model,
            messages=[{"role": "user", "content": gate_prompt}],
            temperature=0.1
        )

        text = (raw or "")
        yes_count = len(YES_RE.findall(text))
        no_count = len(NO_RE.findall(text))

        # Prefer the LAST explicit YES/NO token (models sometimes mention both in the reason).
        matches = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
        last_token = matches[-1].group(1).upper() if matches else ""
        if last_token == "YES":
            return {
                "respond": True, "raw": raw, "decision": "YES_LAST_TOKEN",
                "prompt_preview": prompt_preview, "yes_count": yes_count, "no_count": no_count
            }
        if last_token == "NO":
            return {
                "respond": False, "raw": raw, "decision": "NO_LAST_TOKEN",
                "prompt_preview": prompt_preview, "yes_count": yes_count, "no_count": no_count
            }

        if yes_count == 0 and no_count == 0:
            return {
                "respond": False, "raw": raw, "decision": "NO_KEYWORDS",
                "prompt_preview": prompt_preview, "yes_count": 0, "no_count": 0
            }

        if yes_count == no_count:
            return {
                "respond": False, "raw": raw, "decision": "TIE",
                "prompt_preview": prompt_preview, "yes_count": yes_count, "no_count": no_count
            }

        if yes_count > no_count:
            return {
                "respond": True, "raw": raw, "decision": "YES_BY_COUNT",
                "prompt_preview": prompt_preview, "yes_count": yes_count, "no_count": no_count
            }

        return {
            "respond": False, "raw": raw, "decision": "NO_BY_COUNT",
            "prompt_preview": prompt_preview, "yes_count": yes_count, "no_count": no_count
        }

    except Exception as e:
        last = msgs[-1]
        respond = last.author_type == "user"
        return {
            "respond": respond,
            "raw": f"[gate_error] {e}",
            "decision": "FALLBACK_LAST_IS_USER",
            "prompt_preview": prompt_preview,
            "yes_count": 0,
            "no_count": 0
        }


# -------------------------
# Message helpers
# -------------------------
def _find_message(conversation_id: str, message_id: str) -> Tuple[Optional[Message], Optional[int]]:
    msgs = MESSAGES.get(conversation_id, [])
    for i, m in enumerate(msgs):
        if m.id == message_id:
            return m, i
    return None, None


async def _broadcast_message_update(conversation_id: str, msg: Message):
    await ws_manager.broadcast(conversation_id, {"type": "message_update", "data": msg.model_dump()})


async def _maybe_humanize_delay(character: Character, reply_text: str):
    if not character.humanize_pacing:
        return
    # Small jitter before sending, plus typing time proportional to length (capped)
    pre = random.uniform(0.15, 0.55)
    per_char = random.uniform(0.010, 0.022)
    t = min(max(pre + (len(reply_text) * per_char), 0.4), 4.0)
    await asyncio.sleep(t)


# -------------------------
# Character response (with dev “draft” event)
# -------------------------
async def character_respond(conversation_id: str, character: Character):
    if CHAR_BUSY.get(character.id, False):
        return
    if PAUSE_ALL_CALLS:
        return

    responded_to_ts = _latest_nonself_message_ts(conversation_id, character.id)
    latest_msg_ts = _latest_message_ts(conversation_id)
    last_handled = LAST_RESPONDED_TO.get(conversation_id, {}).get(character.id, 0.0)
    idle_s = max(0, time.time() - latest_msg_ts) if latest_msg_ts else 0.0
    idle_nudge = (IDLE_NUDGE_SECONDS > 0) and (latest_msg_ts > 0) and (idle_s >= IDLE_NUDGE_SECONDS) and (responded_to_ts <= last_handled)

    # Extra cooldown to prevent accidental double-sends.
    last_active = float(CHAR_LAST_ACTIVE.get(character.id, 0.0) or 0.0)
    if last_active and (time.time() - last_active) < float(MIN_SECONDS_BETWEEN_CHARACTER_MESSAGES):
        await debug_event(conversation_id, 'info', 'generate_skip_cooldown', {
            'character': {'id': character.id, 'name': character.name},
            'cooldown_s': float(MIN_SECONDS_BETWEEN_CHARACTER_MESSAGES)
        })
        return

    CHAR_BUSY[character.id] = True
    try:
        await ws_manager.broadcast(conversation_id, {
            "type": "typing",
            "data": {"character_id": character.id, "name": character.name, "typing": True, "ts": time.time()}
        })

        await debug_event(conversation_id, "info", "generate_start", {
            "character": {"id": character.id, "name": character.name},
            "provider": _norm_provider(character.provider),
            "provider_label": _provider_label(character.provider),
            "model": character.model
        })

        # BUILD CONTEXT ONCE
        ctx = build_context_for_character(conversation_id, character)

        ctx_preview_lines = []
        for item in ctx[-8:]:
            ctx_preview_lines.append(f"{item['role']}: {item['content']}")
        await debug_event(conversation_id, "info", "chat_prompt_preview", {
            "character": {"id": character.id, "name": character.name},
            "provider": _norm_provider(character.provider),
            "provider_label": _provider_label(character.provider),
            "model": character.model,
            "preview": _truncate("\n".join(ctx_preview_lines), 900)
        })

        reply = await llm_chat(provider=character.provider, model=character.model, messages=ctx, temperature=2)
        reply = (reply or "").strip()

        # If the model repeats itself, retry once with a strong anti-repeat hint.
        last_self = _last_message_by_character(conversation_id, character.id)
        if last_self and _norm_text(reply) == _norm_text(last_self):
            await debug_event(conversation_id, 'warn', 'duplicate_reply_retry', {
                'character': {'id': character.id, 'name': character.name},
            })
            ctx2 = list(ctx) + [{
                'role': 'system',
                'content': 'Do NOT repeat your previous message. Say something new, or ask a fresh question. Avoid reusing the same first sentence.'
            }]
            reply2 = await llm_chat(provider=character.provider, model=character.model, messages=ctx2, temperature=2)
            reply2 = (reply2 or '').strip()
            if reply2 and _norm_text(reply2) != _norm_text(last_self):
                reply = reply2
            else:
                # still a duplicate → skip sending
                await debug_event(conversation_id, 'warn', 'duplicate_reply_suppressed', {
                    'character': {'id': character.id, 'name': character.name}
                })
                return

        await debug_event(conversation_id, "info", "draft_reply", {
            "character": {"id": character.id, "name": character.name},
            "draft": _truncate(reply, 1200)
        })

        sent_any = False
        if reply:
            await _maybe_humanize_delay(character, reply)

            msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                author_type="character",
                author_id=character.id,
                content=reply,
                ts=time.time(),
            )
            MESSAGES.setdefault(conversation_id, []).append(msg)
            CHAR_LAST_ACTIVE[character.id] = msg.ts

            await ws_manager.broadcast(conversation_id, {"type": "message", "data": msg.model_dump()})

            await debug_event(conversation_id, "info", "message_committed", {
                "character": {"id": character.id, "name": character.name},
                "message_id": msg.id
            })

            sent_any = True

        if sent_any:
            LAST_RESPONDED_TO.setdefault(conversation_id, {})[character.id] = responded_to_ts
            if idle_nudge:
                LAST_IDLE_NUDGE.setdefault(conversation_id, {})[character.id] = latest_msg_ts
            _save_state()

    except Exception as e:
        await debug_event(conversation_id, "error", "generate_error", {
            "character": {"id": character.id, "name": character.name},
            "error": str(e)
        })
    finally:
        CHAR_BUSY[character.id] = False
        await ws_manager.broadcast(conversation_id, {
            "type": "typing",
            "data": {"character_id": character.id, "name": character.name, "typing": False, "ts": time.time()}
        })


# -------------------------
# Presence loop (cosmetic)
# -------------------------
def _compute_presence_status(char: Character) -> str:
    if not char.enabled:
        return "offline"
    if CHAR_BUSY.get(char.id, False):
        return "typing"
    last = CHAR_LAST_ACTIVE.get(char.id, 0.0)
    if last and (time.time() - last) > float(max(10, char.idle_after_seconds)):
        return "idle"
    return "online"


def _presence_payload_for_conversation(conversation_id: str) -> List[dict]:
    conv = CONVERSATIONS.get(conversation_id)
    if not conv:
        return []
    out = []
    for cid in conv.character_ids:
        c = CHARACTERS.get(cid)
        if not c:
            continue
        out.append({
            "id": c.id,
            "name": c.name,
            "enabled": c.enabled,
            "status": _compute_presence_status(c),
            "about": c.about,
            "profile_color": c.profile_color,
        })
    return out


async def presence_loop():
    while True:
        try:
            for conv_id in list(CONVERSATIONS.keys()):
                await ws_manager.broadcast(conv_id, {
                    "type": "presence",
                    "data": {"conversation_id": conv_id, "members": _presence_payload_for_conversation(conv_id), "ts": time.time()}
                })
        except Exception:
            pass
        await asyncio.sleep(PRESENCE_TICK_SECONDS)


# -------------------------
# Background gate loop
# -------------------------
async def gate_loop():
    while True:
        try:
            now = time.time()

            for conv_id, conv in list(CONVERSATIONS.items()):
                LAST_GATE_CHECK.setdefault(conv_id, {})

                for char_id in list(conv.character_ids):
                    if PAUSE_ALL_CALLS:
                        continue
                    char = CHARACTERS.get(char_id)
                    if not char or not char.enabled:
                        continue

                    interval = max(1, int(char.gate_interval_seconds))
                    last_check = LAST_GATE_CHECK[conv_id].get(char_id, 0.0)
                    if now - last_check < interval:
                        continue
                    LAST_GATE_CHECK[conv_id][char_id] = now

                    await debug_event(conv_id, "info", "gate_check_start", {
                        "character": {"id": char.id, "name": char.name},
                        "gate_provider": _norm_provider(char.gate_provider),
                        "gate_provider_label": _provider_label(char.gate_provider),
                        "gate_model": char.gate_model,
                        "interval_s": char.gate_interval_seconds
                    })

                    if _norm_provider(char.gate_provider) == "ollama":
                        _maybe_start_ollama()
                    if _norm_provider(char.gate_provider) == "ollama" and not ollama_is_up():
                        await debug_event(conv_id, "warn", "ollama_offline", {
                            "note": "Ollama not responding; gate check skipped (gate_provider=ollama)."
                        })
                        continue

                    latest_nonself_ts = _latest_nonself_message_ts(conv_id, char.id)
                    last_handled = LAST_RESPONDED_TO.get(conv_id, {}).get(char.id, 0.0)
                    latest_msg_ts = _latest_message_ts(conv_id)
                    idle_s = max(0, now - latest_msg_ts) if latest_msg_ts else 0.0
                    last_idle = LAST_IDLE_NUDGE.get(conv_id, {}).get(char.id, 0.0)
                    idle_eligible = (
                        IDLE_NUDGE_SECONDS > 0
                        and latest_msg_ts > 0
                        and idle_s >= IDLE_NUDGE_SECONDS
                        and last_idle < latest_msg_ts
                    )
                    no_new_activity = latest_nonself_ts <= last_handled and not idle_eligible

                    # Anti-runaway: if the conversation is producing too many AI messages too quickly, pause auto replies.
                    last_msg = (MESSAGES.get(conv_id) or [])[-1] if MESSAGES.get(conv_id) else None
                    if last_msg and last_msg.author_type == "character" and _auto_rate_limited(conv_id):
                        await debug_event(conv_id, "warn", "gate_rate_limited", {
                            "character": {"id": char.id, "name": char.name},
                            "limit_per_min": int(MAX_AUTO_MESSAGES_PER_MINUTE)
                        })
                        continue

                    # NOTE: Gate raw output is not stored in messages.
                    gate = await judge_should_respond(conv_id, char)

                    await debug_event(conv_id, "info", "gate_result", {
                        "character": {"id": char.id, "name": char.name},
                        "gate_provider": _norm_provider(char.gate_provider),
                        "gate_provider_label": _provider_label(char.gate_provider),
                        "gate_model": char.gate_model,
                        "decision": gate.get("decision"),
                        "respond": gate.get("respond"),
                        "yes_count": gate.get("yes_count"),
                        "no_count": gate.get("no_count"),
                        "raw": _truncate(gate.get("raw", ""), 400),
                        "prompt_preview": gate.get("prompt_preview", "")
                    })

                    if gate.get("respond"):
                        await character_respond(conv_id, char)

        except Exception as e:
            print("[WARN] gate_loop error:", e)

        await asyncio.sleep(1)


# -------------------------
# FastAPI app
# -------------------------
@asynccontextmanager
def _any_uses_ollama() -> bool:
    for c in CHARACTERS.values():
        if _norm_provider(c.provider) == "ollama" or _norm_provider(c.gate_provider) == "ollama":
            return True
    return False


async def lifespan(app: FastAPI):

    # Load persisted state first (if any)
    _load_state()

    # Seed starter character + conversation
    if not CHARACTERS:
        nova = Character(
            id=str(uuid.uuid4()),
            name="Nova",
            provider=_norm_provider(LLM_PROVIDER_DEFAULT),
            model=(
                "mistralai/devstral-2512:free" if _norm_provider(LLM_PROVIDER_DEFAULT) == "openrouter"
                else (PUTER_DEFAULT_MODEL or CHAT_MODEL_DEFAULT)
                if _norm_provider(LLM_PROVIDER_DEFAULT) == "puter"
                else CHAT_MODEL_DEFAULT
            ),
            system_prompt="You are Nova. Friendly, witty, concise. You chat like a real Discord friend.",
            gate_provider=_norm_provider(LLM_PROVIDER_DEFAULT),
            gate_model=(
                "mistralai/devstral-2512:free" if _norm_provider(LLM_PROVIDER_DEFAULT) == "openrouter"
                else (PUTER_DEFAULT_MODEL or GATE_MODEL_DEFAULT)
                if _norm_provider(LLM_PROVIDER_DEFAULT) == "puter"
                else GATE_MODEL_DEFAULT
            ),
            gate_interval_seconds=30,
            enabled=True,
            about="Friendly, witty, concise.",
            humanize_pacing=False,
        )
        CHARACTERS[nova.id] = nova
        CHAR_BUSY[nova.id] = False
        CHAR_LAST_ACTIVE[nova.id] = time.time()
    if not CONVERSATIONS:
        conv = Conversation(
            id=str(uuid.uuid4()),
            title="general",
            character_ids=list(CHARACTERS.keys())[:1],
        )
        CONVERSATIONS[conv.id] = conv
        MESSAGES[conv.id] = []
        LAST_GATE_CHECK[conv.id] = {}
        LAST_RESPONDED_TO[conv.id] = {}
        LAST_IDLE_NUDGE[conv.id] = {}
        _save_state()

    if OLLAMA_START_MODE == "always" and _any_uses_ollama():
        ensure_ollama_running()

    asyncio.create_task(gate_loop())
    asyncio.create_task(presence_loop())
    yield


app = FastAPI(title="Discord AI2026", lifespan=lifespan)


# -------------------------
# UI + helpers
# -------------------------
@app.get("/", response_class=HTMLResponse)
def ui():
    if not os.path.exists(INDEX_PATH):
        return HTMLResponse(
            "<h1>Missing index.html</h1><p>Put index.html in the same folder as main.py</p>",
            status_code=500
        )
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health():
    return {
        "ok": True,
        "ollama_up": ollama_is_up(),
        "ollama_start_mode": OLLAMA_START_MODE,
        "openrouter_configured": openrouter_configured(),
        "puter_configured": puter_configured(),
        "llm_provider_default": _norm_provider(LLM_PROVIDER_DEFAULT),
        "defaults": {
            "chat_model": CHAT_MODEL_DEFAULT,
            "gate_model": GATE_MODEL_DEFAULT,
            "openrouter_default_model": OPENROUTER_DEFAULT_MODEL,
            "puter_default_model": PUTER_DEFAULT_MODEL,
        },
        "counts": {"conversations": len(CONVERSATIONS), "characters": len(CHARACTERS)},
    }


@app.get("/user/profile")
def get_user_profile():
    return {"username": _resolve_user_display("user")}


@app.post("/user/profile")
def set_user_profile(payload: UserProfile):
    name = _normalize_username(payload.username)
    USER_PROFILE["username"] = name
    _save_state()
    return {"username": name}


class OllamaStartMode(BaseModel):
    mode: str


@app.get("/settings/ollama_start")
def get_ollama_start():
    return {"mode": OLLAMA_START_MODE}


@app.post("/settings/ollama_start")
def set_ollama_start(payload: OllamaStartMode):
    _set_ollama_start_mode(payload.mode)
    _save_state()
    return {"mode": OLLAMA_START_MODE}


@app.get("/debug/pause")
def get_pause():
    return {"enabled": bool(PAUSE_ALL_CALLS)}


@app.post("/debug/pause")
def set_pause(payload: PauseToggle):
    global PAUSE_ALL_CALLS
    PAUSE_ALL_CALLS = bool(payload.enabled)
    return {"enabled": PAUSE_ALL_CALLS}



@app.get("/openrouter/health")
def openrouter_health():
    return {
        "ok": True,
        "configured": openrouter_configured(),
        "base_url": OPENROUTER_BASE_URL,
        "default_model": OPENROUTER_DEFAULT_MODEL,
        "note": "Set OPENROUTER_API_KEY to enable OpenRouter calls.",
    }


@app.get("/ollama/models")
async def list_ollama_models():
    if not ollama_is_up():
        return {"ok": False, "models": [], "error": "Ollama is offline"}
    try:
        models = await ollama_tags()
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


@app.get("/debug/state")
def debug_state():
    return {
        "characters": [c.model_dump() for c in CHARACTERS.values()],
        "conversations": [c.model_dump() for c in CONVERSATIONS.values()],
        "last_gate_check": LAST_GATE_CHECK,
        "char_busy": CHAR_BUSY,
        "pause_all_calls": PAUSE_ALL_CALLS,
    }


# -------------------------
# REST API
# -------------------------
@app.get("/characters", response_model=List[Character])
def list_characters():
    return list(CHARACTERS.values())


@app.post("/characters", response_model=Character)
def create_character(payload: CharacterCreate):
    cid = str(uuid.uuid4())
    data = payload.model_dump()
    data["provider"] = _norm_provider(data.get("provider"))
    data["gate_provider"] = _norm_provider(data.get("gate_provider") or data.get("provider"))

    # Sensible defaults when using OpenRouter
    if data["provider"] == "openrouter" and not (data.get("model") or "").strip():
        data["model"] = OPENROUTER_DEFAULT_MODEL
    if data["gate_provider"] == "openrouter" and not (data.get("gate_model") or "").strip():
        data["gate_model"] = OPENROUTER_DEFAULT_MODEL
    if data["provider"] == "puter" and not (data.get("model") or "").strip():
        data["model"] = PUTER_DEFAULT_MODEL
    if data["gate_provider"] == "puter" and not (data.get("gate_model") or "").strip():
        data["gate_model"] = PUTER_DEFAULT_MODEL
    if (data["provider"] == "openrouter" or data["gate_provider"] == "openrouter") and not openrouter_configured():
        raise HTTPException(status_code=400, detail="OpenRouter API key missing. Set OPENROUTER_API_KEY.")

    c = Character(id=cid, **data)
    CHARACTERS[cid] = c
    CHAR_BUSY[cid] = False
    CHAR_LAST_ACTIVE[cid] = time.time()

    # Auto-join first conversation so it "just works"
    if CONVERSATIONS:
        first_conv_id = next(iter(CONVERSATIONS.keys()))
        conv = CONVERSATIONS[first_conv_id]
        if cid not in conv.character_ids:
            conv.character_ids.append(cid)
            CONVERSATIONS[first_conv_id] = conv

    _save_state()
    return c


@app.patch("/characters/{character_id}", response_model=Character)
def update_character(character_id: str, payload: CharacterUpdate):
    if character_id not in CHARACTERS:
        raise HTTPException(status_code=404, detail="Character not found")
    c = CHARACTERS[character_id]
    data = c.model_dump()
    patch = payload.model_dump(exclude_none=True)
    data.update(patch)

    # Normalize providers + apply defaults
    data["provider"] = _norm_provider(data.get("provider"))
    data["gate_provider"] = _norm_provider(data.get("gate_provider") or data.get("provider"))

    if data["provider"] == "openrouter" and not (data.get("model") or "").strip():
        data["model"] = OPENROUTER_DEFAULT_MODEL
    if data["gate_provider"] == "openrouter" and not (data.get("gate_model") or "").strip():
        data["gate_model"] = OPENROUTER_DEFAULT_MODEL
    if data["provider"] == "puter" and not (data.get("model") or "").strip():
        data["model"] = PUTER_DEFAULT_MODEL
    if data["gate_provider"] == "puter" and not (data.get("gate_model") or "").strip():
        data["gate_model"] = PUTER_DEFAULT_MODEL
    if (data["provider"] == "openrouter" or data["gate_provider"] == "openrouter") and not openrouter_configured():
        raise HTTPException(status_code=400, detail="OpenRouter API key missing. Set OPENROUTER_API_KEY.")

    updated = Character(**data)
    CHARACTERS[character_id] = updated
    if character_id not in CHAR_BUSY:
        CHAR_BUSY[character_id] = False
    if character_id not in CHAR_LAST_ACTIVE:
        CHAR_LAST_ACTIVE[character_id] = time.time()
    _save_state()
    return updated


@app.get("/conversations", response_model=List[Conversation])
def list_conversations():
    return list(CONVERSATIONS.values())


@app.post("/conversations", response_model=Conversation)
def create_conversation(payload: ConversationCreate):
    cid = str(uuid.uuid4())
    conv = Conversation(id=cid, **payload.model_dump())
    CONVERSATIONS[cid] = conv
    MESSAGES.setdefault(cid, [])
    LAST_GATE_CHECK.setdefault(cid, {})
    LAST_RESPONDED_TO.setdefault(cid, {})
    LAST_IDLE_NUDGE.setdefault(cid, {})
    _save_state()
    return conv


@app.patch("/conversations/{conversation_id}", response_model=Conversation)
def update_conversation(conversation_id: str, payload: ConversationUpdate):
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = CONVERSATIONS[conversation_id]
    data = conv.model_dump()
    patch = payload.model_dump(exclude_none=True)
    data.update(patch)
    updated = Conversation(**data)
    CONVERSATIONS[conversation_id] = updated
    MESSAGES.setdefault(conversation_id, [])
    LAST_GATE_CHECK.setdefault(conversation_id, {})
    LAST_RESPONDED_TO.setdefault(conversation_id, {})
    LAST_IDLE_NUDGE.setdefault(conversation_id, {})
    _save_state()
    return updated


@app.get("/conversations/{conversation_id}/messages", response_model=List[Message])
def get_messages(conversation_id: str):
    return MESSAGES.get(conversation_id, [])


@app.post("/conversations/{conversation_id}/messages", response_model=Message)
async def post_message(conversation_id: str, payload: MessageCreate):
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")

    author_id = payload.author_id
    if payload.author_type == "user":
        if not (author_id or "").strip() or (author_id or "").strip().lower() == "user":
            author_id = "user"
        else:
            author_id = _normalize_username(author_id)

    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        author_type=payload.author_type,
        author_id=author_id,
        content=payload.content,
        reply_to=payload.reply_to,
        ts=time.time(),
    )
    MESSAGES.setdefault(conversation_id, []).append(msg)

    if payload.author_type == "character":
        CHAR_LAST_ACTIVE[payload.author_id] = msg.ts
    elif payload.author_type == "user":
        conv = CONVERSATIONS.get(conversation_id)
        if conv:
            for cid in conv.character_ids:
                LAST_GATE_CHECK.setdefault(conversation_id, {})[cid] = 0.0

    _save_state()
    await ws_manager.broadcast(conversation_id, {"type": "message", "data": msg.model_dump()})
    return msg


@app.patch("/conversations/{conversation_id}/messages/{message_id}", response_model=Message)
async def edit_message(conversation_id: str, message_id: str, payload: MessageEdit):
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg, idx = _find_message(conversation_id, message_id)
    if not msg or idx is None:
        raise HTTPException(status_code=404, detail="Message not found")

    # simple permission: only author can edit
    if msg.author_type == "user":
        if payload.editor_type != "user":
            raise HTTPException(status_code=403, detail="Not allowed to edit this message")
    elif not (payload.editor_type == msg.author_type and payload.editor_id == msg.author_id):
        raise HTTPException(status_code=403, detail="Not allowed to edit this message")

    updated = msg.model_copy(update={
        "content": payload.content,
        "edited_ts": time.time(),
    })
    MESSAGES[conversation_id][idx] = updated
    await _broadcast_message_update(conversation_id, updated)
    _save_state()
    return updated


@app.post("/conversations/{conversation_id}/messages/{message_id}/reactions", response_model=Message)
async def toggle_reaction(conversation_id: str, message_id: str, payload: ReactionToggle):
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg, idx = _find_message(conversation_id, message_id)
    if not msg or idx is None:
        raise HTTPException(status_code=404, detail="Message not found")

    reactions = dict(msg.reactions or {})
    actors = reactions.get(payload.emoji, [])
    if payload.actor_id in actors:
        actors = [a for a in actors if a != payload.actor_id]
    else:
        actors = actors + [payload.actor_id]
    if actors:
        reactions[payload.emoji] = actors
    else:
        reactions.pop(payload.emoji, None)

    updated = msg.model_copy(update={"reactions": reactions})
    MESSAGES[conversation_id][idx] = updated
    await _broadcast_message_update(conversation_id, updated)
    _save_state()
    return updated


@app.post("/conversations/{conversation_id}/messages/{message_id}/pin", response_model=Message)
async def toggle_pin(conversation_id: str, message_id: str, payload: PinToggle):
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg, idx = _find_message(conversation_id, message_id)
    if not msg or idx is None:
        raise HTTPException(status_code=404, detail="Message not found")

    new_pinned = (not msg.pinned) if payload.pinned is None else bool(payload.pinned)
    updated = msg.model_copy(update={"pinned": new_pinned})
    MESSAGES[conversation_id][idx] = updated
    await _broadcast_message_update(conversation_id, updated)
    _save_state()
    return updated


@app.get("/conversations/{conversation_id}/pins", response_model=List[Message])
def list_pins(conversation_id: str):
    msgs = MESSAGES.get(conversation_id, [])
    return [m for m in msgs if m.pinned]


@app.get("/conversations/{conversation_id}/search", response_model=List[Message])
def search_messages(conversation_id: str, q: str = "", limit: int = 50):
    q = (q or "").strip().lower()
    if not q:
        return []
    msgs = MESSAGES.get(conversation_id, [])
    out = []
    for m in reversed(msgs):
        if q in (m.content or "").lower():
            out.append(m)
            if len(out) >= int(max(1, min(200, limit))):
                break
    return out


# -------------------------
# WebSocket
# -------------------------
@app.websocket("/ws/{conversation_id}")
async def ws_conversation(ws: WebSocket, conversation_id: str):
    await ws_manager.join(conversation_id, ws)
    try:
        await ws.send_text(json.dumps({
            "type": "init",
            "data": {
                "conversation_id": conversation_id,
                "messages": [m.model_dump() for m in MESSAGES.get(conversation_id, [])],
                "members": _presence_payload_for_conversation(conversation_id),
            }
        }))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if msg.get("type") == "set_debug":
                ws_manager.set_debug(ws, bool(msg.get("enabled", False)))
                await ws.send_text(json.dumps({
                    "type": "debug_ack",
                    "data": {"enabled": bool(msg.get("enabled", False))}
                }))
            elif msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong", "data": {"ts": time.time()}}))

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.leave(conversation_id, ws)


# -------------------------
# Run: python main.py
# -------------------------
def _open_browser_once():
    time.sleep(1.2)
    try:
        webbrowser.open(f"http://{APP_HOST}:{APP_PORT}/")
    except Exception:
        pass


if __name__ == "__main__":
    print(f"[BOOT] Folder: {HERE}")
    print(f"[BOOT] Defaults: chat={CHAT_MODEL_DEFAULT} gate={GATE_MODEL_DEFAULT}")
    threading.Thread(target=_open_browser_once, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
