# /app/bridge_server.py
import os
import re
import json
import asyncio
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from difflib import get_close_matches
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query
from fastapi.middleware.cors import CORSMiddleware

# ============================
# Local imports
# ============================
from managers.db_manager import insert_repo_to_db, get_connection
from managers.prompt_agent import LLMAgent
from managers.llm_manager import LLMManager
from managers.context_manager import ContextManager
from utils.torch_version_loader import TorchVersionLoader


# ============================
# Base setup
# ============================
BASE_DIR = Path(__file__).parent.resolve()
GIT_CLONE_DIR = (BASE_DIR / "workspace").resolve()
GIT_CLONE_DIR.mkdir(parents=True, exist_ok=True)
BRIDGE_PORT = 9013

app = FastAPI(title="Bridge Server (React ↔ FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Global managers & locks
# ============================
clients: List[WebSocket] = []
clients_lock = asyncio.Lock()
llm_lock = asyncio.Lock()
llm_manager = LLMManager()
agent = LLMAgent(llm_manager)
torch_loader = TorchVersionLoader(base_dir="/app/pytorch_versions")
context_manager = ContextManager()
RG_BINARY = shutil.which("rg")


# ============================
# Utility Functions
# ============================
def _normalize_trigger_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


CLOSING_BEHAVIOR_TRIGGER = (
    llm_manager.prompts.get("code", {}).get("closing_behavior_trigger")
    or "Would you like me to refactor this for the latest PyTorch version, or a specific version you prefer?"
)
CLOSING_BEHAVIOR_TRIGGER_NORMALIZED = _normalize_trigger_text(CLOSING_BEHAVIOR_TRIGGER)
LAST_CODE_REQUESTS: Dict[Any, Dict[str, Any]] = {}


def _tab_storage_key(tab_id: int | None):
    return tab_id if tab_id is not None else "__default__"


def _closing_trigger_matches(text: str | None) -> bool:
    if not text or not CLOSING_BEHAVIOR_TRIGGER_NORMALIZED:
        return False
    return _normalize_trigger_text(text) == CLOSING_BEHAVIOR_TRIGGER_NORMALIZED


def _remember_code_request(tab_id: int | None, user_text: str, response_text: str | None):
    LAST_CODE_REQUESTS[_tab_storage_key(tab_id)] = {
        "user_text": user_text,
        "response_text": response_text,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _get_last_code_request(tab_id: int | None):
    return LAST_CODE_REQUESTS.get(_tab_storage_key(tab_id))


# ============================
# Intent Routing
# ============================
def load_intent_routing():
    fallback = (
        "해당 요청은 아직 전용 에이전트가 준비되지 않았습니다. "
        "추가 지침을 제공해 주시면 다른 방식으로 도와볼게요."
    )
    config = llm_manager.prompts.get("intent_routing", {})
    labels = set(config.get("labels") or [])
    tasks = config.get("response_tasks") or {}
    cfg_fallback = config.get("fallback") or fallback

    if not labels and tasks:
        labels = set(tasks.keys())

    return labels or {"chat", "code", "file", "explain"}, tasks or {}, cfg_fallback


INTENT_LABELS, INTENT_RESPONSE_TASKS, DEFAULT_RESPONSE_FALLBACK = load_intent_routing()


async def broadcast(msg: Dict[str, Any]):
    print(f"[Bridge] 📨 {msg}")
    dead = []
    async with clients_lock:
        for ws in clients:
            try:
                await ws.send_json(json.loads(json.dumps(msg, default=str)))
            except Exception:
                dead.append(ws)
        for d in dead:
            clients.remove(d)


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


# ============================
# LLM Handling
# ============================
def _strip_reasoning_output(text: str) -> str:
    """Remove <think>...</think> reasoning traces."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def run_llm_call(
    prompt: str,
    *,
    task: str,
    max_new_tokens: int = 512,
    system_override: str | None = None,
) -> str:
    async with llm_lock:
        result = await asyncio.to_thread(
            llm_manager.generate,
            prompt,
            task=task,
            max_new_tokens=max_new_tokens,
            system_override=system_override,
        )
        return _strip_reasoning_output(result)


def normalize_intent_label(raw_label: str) -> str:
    label = raw_label.strip().lower()
    if label in INTENT_LABELS:
        return label
    tokens = re.findall(r"[a-z_]+", label)
    for token in tokens:
        if token in INTENT_LABELS:
            return token
    match = get_close_matches(label, INTENT_LABELS, n=1, cutoff=0.6)
    return match[0] if match else "explain"


async def classify_intent(user_text: str) -> str:
    raw = await run_llm_call(user_text, task="classifier_intent", max_new_tokens=16)
    return normalize_intent_label(raw)


async def generate_intent_response(full_prompt: str, intent: str) -> tuple[str | None, str]:
    """Run LLM using the full conversation context prompt."""
    task = INTENT_RESPONSE_TASKS.get(intent)
    if not task:
        return None, DEFAULT_RESPONSE_FALLBACK
    response = await run_llm_call(full_prompt, task=task)
    return task, response


# ============================
# User Message Handling
# ============================
async def handle_user_message(user_text: str, tab_id: int | None):
    try:
        # 1️⃣ 사용자 입력 저장
        context_manager.add_message(tab_id, "user", user_text)

        # 2️⃣ closing behavior 트리거 체크
        if _closing_trigger_matches(user_text):
            await handle_closing_behavior_request(tab_id)
            return

        # 3️⃣ intent 분류
        intent = await classify_intent(user_text)
        await broadcast(
            {
                "type": "intent_detected",
                "text": f"Intent classified: {intent}",
                "data": {"intent": intent},
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )

        # 4️⃣ 전체 문맥 기반 프롬프트 구성
        full_prompt = context_manager.build_prompt(tab_id, user_text)

        # 5️⃣ LLM 호출 (문맥 포함)
        task_used, response_text = await generate_intent_response(full_prompt, intent)

        # 6️⃣ reasoning 제거 후 저장
        clean_response = _strip_reasoning_output(response_text)
        context_manager.add_message(tab_id, "assistant", clean_response)

        # 7️⃣ 응답 브로드캐스트
        await broadcast(
            {
                "type": "llm_response" if task_used else "intent_notice",
                "text": clean_response,
                "data": {"intent": intent, "task": task_used},
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )

        # 8️⃣ code 요청일 경우 마지막 요청 저장
        if task_used == "code":
            _remember_code_request(tab_id, user_text, clean_response)

    except Exception as exc:
        await broadcast(
            {
                "type": "error",
                "text": f"LLM 처리 중 오류가 발생했습니다: {exc}",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        print(f"[Bridge] ⚠️ handle_user_message failed: {exc}")


# ============================
# Git & Repo Handling
# ============================
def extract_github_url(text: str) -> str | None:
    match = re.search(r"(https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
    return match.group(1) if match else None


IGNORE_DIRS = {".git", "venv", "__pycache__", "node_modules"}


def build_dir_tree(base_path: Path, root_path: Path | None = None, max_depth: int = 5, depth: int = 0):
    if root_path is None:
        root_path = base_path
    tree = {"name": base_path.name, "path": str(base_path.relative_to(GIT_CLONE_DIR)), "type": "folder", "children": []}
    if depth > max_depth or not base_path.is_dir():
        return tree
    for entry in sorted(base_path.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        if entry.name in IGNORE_DIRS:
            continue
        if entry.is_dir():
            tree["children"].append(build_dir_tree(entry, root_path, max_depth, depth + 1))
        else:
            tree["children"].append({"name": entry.name, "path": str(entry.relative_to(GIT_CLONE_DIR)), "type": "file"})
    return tree


async def clone_repo_and_broadcast(url: str):
    repo_name = url.split("/")[-1].replace(".git", "")
    dest = GIT_CLONE_DIR / repo_name

    if dest.exists():
        await asyncio.to_thread(subprocess.run, ["git", "-C", str(dest), "pull"], check=True)
    else:
        await asyncio.to_thread(subprocess.run, ["git", "clone", url, str(dest)], check=True)

    repo_id = await asyncio.to_thread(insert_repo_to_db, repo_name, url, dest)
    await broadcast({"type": "git_status", "text": "Summarizing files..."})
    await asyncio.to_thread(agent.summarize_repo_files, repo_id, dest)

    await broadcast({"type": "git_status", "text": "Generating chunks..."})
    await asyncio.to_thread(agent.chunk_repo_files, repo_id, dest)

    await broadcast({"type": "git_status", "text": "Extracting symbol links..."})
    await asyncio.to_thread(agent.extract_symbol_links, repo_id, dest)

    await broadcast({"type": "git_status", "text": "✅ Done."})


# ============================
# FastAPI Routes
# ============================
@app.on_event("startup")
async def startup_event():
    print("========== DEBUG PATH CHECK ==========")
    print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
    print(f"[DEBUG] GIT_CLONE_DIR: {GIT_CLONE_DIR}")
    print(f"[DEBUG] Exists(GIT_CLONE_DIR): {GIT_CLONE_DIR.exists()}")
    print("======================================")


@app.post("/reset_db")
async def reset_db():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            TRUNCATE TABLE repo_meta, files_meta, repo_chunks, symbol_links
            RESTART IDENTITY CASCADE;
        """)
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "ok", "message": "All tables truncated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/send")
async def from_react(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text", "")
    msg_type = payload.get("type", "")
    tab_id = payload.get("tabId")
    github_url = extract_github_url(text)

    if github_url:
        asyncio.create_task(clone_repo_and_broadcast(github_url))
        return {"status": "ok", "message": "Repository cloning and analysis started."}

    if msg_type == "user_input" and text.strip():
        asyncio.create_task(handle_user_message(text.strip(), tab_id))
        return {"status": "ok", "message": "User message queued for processing."}

    return {"status": "ok", "message": "No actionable content found."}


@app.websocket("/ws/client")
async def ws_client(ws: WebSocket):
    await ws.accept()
    async with clients_lock:
        clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)


@app.get("/init_tree")
async def get_initial_tree():
    if not GIT_CLONE_DIR.exists():
        return {"status": "error", "message": "workspace directory not found"}

    entries = [e for e in GIT_CLONE_DIR.iterdir() if e.name not in IGNORE_DIRS]
    if not entries:
        return {"status": "empty", "message": "workspace is empty"}

    trees = [build_dir_tree(e) for e in entries]
    return {"status": "ok", "trees": trees}


@app.get("/file")
async def get_file_content(path: str = Query(...)):
    target = (GIT_CLONE_DIR / path).resolve()
    if not target.exists():
        return {"status": "error", "message": f"file not found: {target}"}
    if target.is_dir():
        return {"status": "error", "message": "cannot open directory"}
    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"status": "error", "message": f"read failed: {e}"}
    return {"status": "ok", "content": content}


@app.get("/history")
async def get_history(limit: int = 100):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT repo_name, repo_url, description, language, total_files, indexed_at
        FROM repo_meta
        ORDER BY indexed_at DESC
        LIMIT %s;
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    history = [
        {
            "repo_name": r[0],
            "repo_url": r[1],
            "description": r[2],
            "language": r[3],
            "total_files": r[4],
            "indexed_at": r[5],
        }
        for r in rows
    ]
    return {"status": "ok", "history": history}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=BRIDGE_PORT, reload=False)
