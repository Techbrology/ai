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

CHAT_MODEL_DEFAULT = os.getenv("CHAT_MODEL", "llama3.2:3b")
GATE_MODEL_DEFAULT = os.getenv("GATE_MODEL", "llama3.2:1b")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_VERSION_ENDPOINT = f"{OLLAMA_BASE_URL}/api/version"
# OpenRouter (optional cloud backend)
LLM_PROVIDER_DEFAULT = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()  # "ollama" | "openrouter"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_CHAT_ENDPOINT = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_API_KEY = os.getenv("api-key-here", "").strip()
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "mistralai/devstral-2512:free").strip()
# Optional metadata headers recommended by OpenRouter
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()   # e.g. "https://example.com"
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Discord AI2026").strip()


def _norm_provider(p: str) -> str:
    p = (p or "").strip().lower()
    if p in ("openrouter", "or", "router", "open-router"):
        return "openrouter"
    return "ollama"


def _provider_label(p: str) -> str:
    p = _norm_provider(p)
    return "OpenRouter" if p == "openrouter" else "Ollama"


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

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # OpenRouter recommends these headers for attribution (optional)
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(OPENROUTER_CHAT_ENDPOINT, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

    try:
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content")
        if content is None and "delta" in choice0:
            content = (choice0.get("delta") or {}).get("content", "")
        return (content or "").strip()
    except Exception:
        return ""


async def llm_chat(provider: str, model: str, messages: List[dict], temperature: float = 0.7) -> str:
    provider = _norm_provider(provider)
    if provider == "openrouter":
        return await openrouter_chat(model=model, messages=messages, temperature=temperature)
    return await ollama_chat(model=model, messages=messages, temperature=temperature)


def openrouter_configured() -> bool:
    return bool(OPENROUTER_API_KEY)




# Presence tuning (purely cosmetic)
PRESENCE_TICK_SECONDS = 5
DEFAULT_IDLE_AFTER_SECONDS = 180


# -------------------------
# Pydantic models
# -------------------------
class CharacterCreate(BaseModel):
    name: str

    # LLM backend selection
    provider: str = Field(default=LLM_PROVIDER_DEFAULT, description="ollama|openrouter")
    model: str = CHAT_MODEL_DEFAULT

    system_prompt: str = "You are a helpful AI."

    # Gate (who decides if the character should speak)
    gate_provider: str = Field(default=LLM_PROVIDER_DEFAULT, description="ollama|openrouter")
    gate_model: str = GATE_MODEL_DEFAULT
    gate_interval_seconds: int = 20

    enabled: bool = True

    # Optional (Discord-like flavor)
    about: str = ""
    profile_color: str = ""  # CSS color (e.g. "#5865F2") or empty for auto
    idle_after_seconds: int = DEFAULT_IDLE_AFTER_SECONDS

    # Optional "human" behavior (OFF by default to preserve old behavior)
    humanize_pacing: bool = False
    split_messages: bool = False


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
    split_messages: Optional[bool] = None


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
    split_messages: bool = False


class ConversationCreate(BaseModel):
    title: str = "general"
    character_ids: List[str] = Field(default_factory=list)


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


# -------------------------
# In-memory state
# -------------------------
CHARACTERS: Dict[str, Character] = {}
CONVERSATIONS: Dict[str, Conversation] = {}
MESSAGES: Dict[str, List[Message]] = {}

LAST_GATE_CHECK: Dict[str, Dict[str, float]] = {}  # conversation_id -> char_id -> ts
CHAR_BUSY: Dict[str, bool] = {}                    # char_id -> busy?

# Cosmetic "last activity" for presence
CHAR_LAST_ACTIVE: Dict[str, float] = {}            # char_id -> ts (message committed)


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

    participants = get_participant_names(conversation_id)
    participants_line = ", ".join(participants) if participants else "(none)"

    system = (
        f"{character.system_prompt}\n\n"
        f"You are '{character.name}' in a group chat.\n"
        f"Participants (AI): {participants_line}\n"
        f"Rules:\n"
        f"- Speak only as {character.name}.\n"
        f"- Do not output meta/debug/gating decisions.\n"
    )

    ctx = [{"role": "system", "content": system}]

    for m in msgs:
        # extra safety: if any gate-ish leak got in somehow, drop it
        leaked = (m.content or "").strip().lower()
        if leaked in ("yes", "no", "the correct answer is yes", "the correct answer is no"):
            continue

        if m.author_type == "user":
            ctx.append({"role": "user", "content": m.content})
        else:
            if m.author_id == character.id:
                ctx.append({"role": "assistant", "content": m.content})
            else:
                other_name = CHARACTERS.get(m.author_id).name if m.author_id in CHARACTERS else "Other"
                ctx.append({"role": "user", "content": f"[{other_name}]: {m.content}"})
    return ctx


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
            transcript_lines.append(f"User: {m.content}")
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

You are deciding if "{character.name}" should reply RIGHT NOW by saying YES or NO at the end of your response. add a reason why you choose your answer at the beginning of your response.

keep your reason short and consice. you MUST use YES or NO in your response.

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


def _split_reply_text(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text or "") if p.strip()]
    if not parts:
        return [(text or "").strip()]
    # avoid turning short replies into many messages
    if len(parts) <= 1:
        return parts
    # cap to prevent spam
    return parts[:5]


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

        reply = await llm_chat(provider=character.provider, model=character.model, messages=ctx, temperature=0.7)
        reply = (reply or "").strip()

        await debug_event(conversation_id, "info", "draft_reply", {
            "character": {"id": character.id, "name": character.name},
            "draft": _truncate(reply, 1200)
        })

        # Optional: send as multiple Discord-like short messages
        chunks = [reply]
        if character.split_messages:
            chunks = _split_reply_text(reply)

        for chunk in chunks:
            chunk = (chunk or "").strip()
            if not chunk:
                continue

            await _maybe_humanize_delay(character, chunk)

            msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                author_type="character",
                author_id=character.id,
                content=chunk,
                ts=time.time(),
            )
            MESSAGES.setdefault(conversation_id, []).append(msg)
            CHAR_LAST_ACTIVE[character.id] = msg.ts

            await ws_manager.broadcast(conversation_id, {"type": "message", "data": msg.model_dump()})

            await debug_event(conversation_id, "info", "message_committed", {
                "character": {"id": character.id, "name": character.name},
                "message_id": msg.id
            })

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
                    char = CHARACTERS.get(char_id)
                    if not char or not char.enabled:
                        continue

                    last_check = LAST_GATE_CHECK[conv_id].get(char_id, 0.0)
                    if now - last_check < char.gate_interval_seconds:
                        continue
                    LAST_GATE_CHECK[conv_id][char_id] = now

                    await debug_event(conv_id, "info", "gate_check_start", {
                        "character": {"id": char.id, "name": char.name},
                        "gate_provider": _norm_provider(char.gate_provider),
                        "gate_provider_label": _provider_label(char.gate_provider),
                        "gate_model": char.gate_model,
                        "interval_s": char.gate_interval_seconds
                    })

                    if _norm_provider(char.gate_provider) == "ollama" and not ollama_is_up():
                        await debug_event(conv_id, "warn", "ollama_offline", {
                            "note": "Ollama not responding; gate check skipped (gate_provider=ollama)."
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
async def lifespan(app: FastAPI):
    ensure_ollama_running()

    # Seed starter character + conversation
# Seed starter character + conversation
    if not CHARACTERS:
        nova = Character(
        id=str(uuid.uuid4()),
        name="Nova",
        provider=_norm_provider(LLM_PROVIDER_DEFAULT),
        model=("mistralai/devstral-2512:free" if _norm_provider(LLM_PROVIDER_DEFAULT)=="openrouter" else CHAT_MODEL_DEFAULT),
        system_prompt="You are Nova. Friendly, witty, concise. You chat like a real Discord friend.",
        gate_provider=_norm_provider(LLM_PROVIDER_DEFAULT),
        gate_model=("mistralai/devstral-2512:free" if _norm_provider(LLM_PROVIDER_DEFAULT)=="openrouter" else GATE_MODEL_DEFAULT),
        gate_interval_seconds=10,
        enabled=True,
        about="Friendly, witty, concise.",
        humanize_pacing=False,
        split_messages=False,
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
        "openrouter_configured": openrouter_configured(),
        "llm_provider_default": _norm_provider(LLM_PROVIDER_DEFAULT),
        "defaults": {
            "chat_model": CHAT_MODEL_DEFAULT,
            "gate_model": GATE_MODEL_DEFAULT,
            "openrouter_default_model": OPENROUTER_DEFAULT_MODEL,
        },
        "counts": {"conversations": len(CONVERSATIONS), "characters": len(CHARACTERS)},
    }



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

    updated = Character(**data)
    CHARACTERS[character_id] = updated
    if character_id not in CHAR_BUSY:
        CHAR_BUSY[character_id] = False
    if character_id not in CHAR_LAST_ACTIVE:
        CHAR_LAST_ACTIVE[character_id] = time.time()
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
    return conv


@app.get("/conversations/{conversation_id}/messages", response_model=List[Message])
def get_messages(conversation_id: str):
    return MESSAGES.get(conversation_id, [])


@app.post("/conversations/{conversation_id}/messages", response_model=Message)
async def post_message(conversation_id: str, payload: MessageCreate):
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        author_type=payload.author_type,
        author_id=payload.author_id,
        content=payload.content,
        reply_to=payload.reply_to,
        ts=time.time(),
    )
    MESSAGES.setdefault(conversation_id, []).append(msg)

    if payload.author_type == "character":
        CHAR_LAST_ACTIVE[payload.author_id] = msg.ts

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
    if not (payload.editor_type == msg.author_type and payload.editor_id == msg.author_id):
        raise HTTPException(status_code=403, detail="Not allowed to edit this message")

    updated = msg.model_copy(update={
        "content": payload.content,
        "edited_ts": time.time(),
    })
    MESSAGES[conversation_id][idx] = updated
    await _broadcast_message_update(conversation_id, updated)
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
