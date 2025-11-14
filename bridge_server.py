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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query
from fastapi.middleware.cors import CORSMiddleware

# ============================
# Local imports
# ============================
from managers.db_manager import insert_repo_to_db, get_connection
from managers.prompt_agent import LLMAgent
from managers.llm_manager import LLMManager
from managers.context_manager import ContextManager
from managers.services.tool_service import ToolService
from managers.services.self_check_service import SelfCheckService
from managers.services.routing_manager import MultiSignalRouter
from managers.embedding import EmbeddingManager
from managers.rag_query import RAGQueryManager
from managers.topic_manager import TopicManager
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
shared_embedder = EmbeddingManager()
context_manager = ContextManager(embedder=shared_embedder)
TOPIC_SIMILARITY_THRESHOLD = 0.6
topic_manager = TopicManager(embedder=shared_embedder, similarity_threshold=TOPIC_SIMILARITY_THRESHOLD)
RG_BINARY = shutil.which("rg")
PRIMARY_TASK = "assistant"
CONTEXT_SIMILARITY_THRESHOLD = 0.55


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
    llm_manager.prompts.get(PRIMARY_TASK, {}).get("closing_behavior_trigger")
    or "Would you like me to refactor this for the latest PyTorch version, or a specific version you prefer?"
)
CLOSING_BEHAVIOR_TRIGGER_NORMALIZED = _normalize_trigger_text(CLOSING_BEHAVIOR_TRIGGER)
LAST_USER_REQUESTS: Dict[Any, Dict[str, Any]] = {}
WORKSPACE_ROOT = Path("/app").resolve()


def _tab_storage_key(tab_id: int | None):
    return tab_id if tab_id is not None else "__default__"


def _closing_trigger_matches(text: str | None) -> bool:
    if not text or not CLOSING_BEHAVIOR_TRIGGER_NORMALIZED:
        return False
    return _normalize_trigger_text(text) == CLOSING_BEHAVIOR_TRIGGER_NORMALIZED


def _remember_user_request(tab_id: int | None, user_text: str, response_text: str | None):
    LAST_USER_REQUESTS[_tab_storage_key(tab_id)] = {
        "user_text": user_text,
        "response_text": response_text,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _get_last_user_request(tab_id: int | None):
    return LAST_USER_REQUESTS.get(_tab_storage_key(tab_id))


def _extract_symbol_candidates(text: str, max_symbols: int = 5) -> List[str]:
    if not text:
        return []
    candidates: List[str] = []
    torch_refs = re.findall(r"(torch(?:\.[A-Za-z_][\w]*)+)", text)
    for ref in torch_refs:
        last = ref.split(".")[-1]
        if last:
            candidates.append(last)
    candidates.extend(re.findall(r"\bdef\s+([A-Za-z_][\w]*)", text))
    candidates.extend(re.findall(r"\bclass\s+([A-Za-z_][\w]*)", text))
    ordered: List[str] = []
    seen = set()
    for name in candidates:
        clean = name.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
        if len(ordered) >= max_symbols:
            break
    return ordered


def _search_symbol_definitions(symbol: str, torch_root: Path, max_matches: int = 1) -> List[tuple[Path, int]]:
    if not symbol or not torch_root.exists() or not RG_BINARY:
        return []
    pattern = rf"^\s*(?:class|def)\s+{re.escape(symbol)}\b"
    cmd = [
        RG_BINARY,
        "--line-number",
        "--no-heading",
        "--max-count",
        str(max_matches),
        "--color",
        "never",
        pattern,
        str(torch_root),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        print(f"[TorchLookup] ⚠️ rg error for symbol {symbol}: {proc.stderr.strip()}")
        return []
    matches: List[tuple[Path, int]] = []
    for line in proc.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        path_str, line_no = parts[0], parts[1]
        try:
            line_idx = int(line_no)
        except ValueError:
            continue
        matches.append((Path(path_str).resolve(), line_idx))
    return matches


def _read_file_snippet(file_path: Path, center_line: int, before: int = 20, after: int = 40) -> str | None:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as exc:
        print(f"[TorchLookup] ⚠️ snippet read failed for {file_path}: {exc}")
        return None
    if not lines:
        return None
    start = max(center_line - 1 - before, 0)
    end = min(center_line - 1 + after, len(lines))
    snippet = "".join(lines[start:end]).strip()
    return snippet or None


def _collect_symbol_snippets(user_text: str, repo_root: Path, max_total: int = 3):
    torch_root = repo_root / "torch"
    symbols = _extract_symbol_candidates(user_text)
    if not torch_root.exists() or not symbols:
        return [], []
    snippets: List[str] = []
    used_symbols: List[str] = []
    for symbol in symbols:
        matches = _search_symbol_definitions(symbol, torch_root, max_matches=1)
        for match_path, line_no in matches:
            try:
                rel_path = match_path.relative_to(repo_root)
            except ValueError:
                rel_path = match_path
            snippet = _read_file_snippet(match_path, line_no)
            if not snippet:
                continue
            block = f"### {rel_path}:{line_no}\n{snippet}"
            snippets.append(block)
            used_symbols.append(symbol)
            if len(snippets) >= max_total:
                return snippets, used_symbols
        if len(snippets) >= max_total:
            break
    return snippets, used_symbols


def _build_torch_source_prompt(user_text: str) -> str | None:
    context = torch_loader.build_context_from_text(user_text)
    if not context:
        return None
    repo_root = context.root_path / "repo"
    if not repo_root.exists():
        return None
    snippets, symbols = _collect_symbol_snippets(user_text, repo_root)
    if not snippets:
        return None
    header = [
        "[Torch Source Lookup]",
        f"Version: {context.version}",
    ]
    if symbols:
        header.append(f"Symbols: {', '.join(symbols)}")
    return "\n".join(header) + "\n\n" + "\n\n".join(snippets)


async def handle_closing_behavior_request(tab_id: int | None):
    last_request = _get_last_user_request(tab_id)
    if not last_request:
        await broadcast(
            {
                "type": "intent_notice",
                "text": "마지막 사용자 요청을 찾을 수 없습니다. 먼저 작업 요청을 전달해 주세요.",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        return

    try:
        torch_prompt = await asyncio.to_thread(_build_torch_source_prompt, last_request["user_text"])
    except Exception as exc:
        await broadcast(
            {
                "type": "error",
                "text": f"PyTorch 소스 탐색 중 오류가 발생했습니다: {exc}",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        print(f"[TorchLookup] ⚠️ build prompt failed: {exc}")
        return

    if not torch_prompt:
        await broadcast(
            {
                "type": "intent_notice",
                "text": "요청과 일치하는 PyTorch 소스 코드를 찾지 못했습니다. 함수명이나 torch.* 경로를 포함해 주세요.",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        return

    conversation_id = topic_manager.active_conversation_id(tab_id) or tab_id
    combined_prompt = f"{torch_prompt}\n\n[User Request]\n{last_request['user_text']}"
    response_text = await run_llm_call(combined_prompt, task=PRIMARY_TASK)
    clean_response = _strip_reasoning_output(response_text)
    context_manager.add_message(conversation_id, "assistant", clean_response)
    active_topic = topic_manager.active_topic_id(tab_id)
    if active_topic:
        topic_embedding = context_manager.get_context_embedding(conversation_id)
        topic_manager.update_topic_embedding(tab_id, active_topic, topic_embedding)
    await broadcast(
        {
            "type": "llm_response",
            "text": clean_response,
            "data": {"task": PRIMARY_TASK, "context": "torch_lookup"},
            "tabId": tab_id,
            "timestamp": current_timestamp(),
        }
    )
    _remember_user_request(tab_id, last_request["user_text"], clean_response)


# ============================
# User Message Handling
# ============================
async def handle_user_message(user_text: str, tab_id: int | None):
    try:
        topic_assignment = topic_manager.assign_topic(tab_id, user_text)
        topic_id = topic_assignment["topic_id"]
        topic_similarity = topic_assignment.get("similarity")
        topic_is_new = topic_assignment.get("is_new")
        conversation_id = topic_manager.topic_conversation_id(tab_id, topic_id)

        include_history = not topic_is_new
        if topic_similarity is not None:
            include_history = topic_similarity >= CONTEXT_SIMILARITY_THRESHOLD
        print(
            f"[Topic] tab={tab_id} topic={topic_id} "
            f"is_new={topic_is_new} similarity={topic_similarity} "
            f"include_history={include_history}"
        )

        context_manager.add_message(conversation_id, "user", user_text)
        tool_service.set_last_user_query(user_text)

        if _closing_trigger_matches(user_text):
            await handle_closing_behavior_request(tab_id)
            return
        self_check = await self_check_service.run(user_text)
        if self_check:
            print(
                f"[SelfCheck] confidence={self_check.get('confidence')} "
                f"freshness={self_check.get('freshness_need')}"
            )

        routing_result = await routing_manager.route(user_text, self_check, state_key=conversation_id)
        function_call = routing_result.get("function_call") or {}
        plan_meta = {
            "name": function_call.get("name"),
            "confidence": function_call.get("confidence"),
            "reason": function_call.get("reason"),
        }

        tool_block = ""
        executed_function = None
        tool_meta = None
        tool_payload = None
        if function_call.get("name") and function_call.get("name") != "none":
            if tool_service.is_answer_direct(function_call):
                executed_function = {"name": "answer_direct", "arguments": {}}
            else:
                try:
                    func_output, executed_function, tool_meta = await tool_service.execute_tool(function_call)
                    tool_block = f"[Tool Result: {executed_function['name']}]\n{func_output}"
                    context_manager.add_message(conversation_id, "tool", tool_block)
                    tool_payload = {
                        "name": executed_function["name"],
                        "arguments": executed_function.get("arguments"),
                        "text": func_output,
                        "meta": tool_meta,
                    }
                    await broadcast(
                        {
                            "type": "tool_result",
                            "text": func_output,
                            "data": {
                                "function": executed_function,
                                "meta": tool_meta,
                            },
                            "tabId": tab_id,
                            "timestamp": current_timestamp(),
                        }
                    )
                except Exception as exc:
                    await broadcast(
                        {
                            "type": "error",
                            "text": f"Function call failed: {exc}",
                            "data": {"function": function_call},
                            "tabId": tab_id,
                            "timestamp": current_timestamp(),
                        }
                    )
                    print(f"[Bridge] ⚠️ function call failed: {exc}")
                    tool_block = ""
                    executed_function = None
                    tool_meta = None

        full_prompt = context_manager.build_prompt(conversation_id, user_text, include_history=include_history)
        if tool_block:
            full_prompt = f"{full_prompt}\n\n{tool_block}"

        response_text = await run_llm_call(full_prompt, task=PRIMARY_TASK)
        clean_response = _strip_reasoning_output(response_text)
        context_manager.add_message(conversation_id, "assistant", clean_response)
        topic_embedding = context_manager.get_context_embedding(conversation_id)
        topic_manager.update_topic_embedding(tab_id, topic_id, topic_embedding)

        context_gate_meta = {
            "includeHistory": include_history,
            "threshold": CONTEXT_SIMILARITY_THRESHOLD,
            "topicSimilarity": topic_similarity,
            "topicId": topic_id,
            "topicIsNew": topic_is_new,
        }
        await broadcast(
            {
                "type": "llm_response",
                "text": clean_response,
                "data": {
                    "task": PRIMARY_TASK,
                    "function": executed_function,
                    "toolResult": tool_payload,
                    "routingDebug": {
                        "functionRouter": routing_result.get("function_router"),
                        "fallback": routing_result.get("fallback_function"),
                        "final": routing_result.get("final_function"),
                    },
                    "planMeta": plan_meta,
                    "routingContext": routing_result.get("routing_context"),
                    "topic": {
                        "id": topic_id,
                        "similarity": topic_similarity,
                        "isNew": topic_is_new,
                        "threshold": TOPIC_SIMILARITY_THRESHOLD,
                    },
                    "contextSimilarity": context_gate_meta,
                    "sourceRouter": routing_result.get("source"),
                    "intentRouter": routing_result.get("intent"),
                    "safetyRouter": routing_result.get("safety"),
                },
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )

        _remember_user_request(tab_id, user_text, clean_response)

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


# ============================
# Services
# ============================
rag_manager = RAGQueryManager(embedder=shared_embedder)
tool_service = ToolService(
    llm_manager=llm_manager,
    run_llm_call=run_llm_call,
    workspace_root=WORKSPACE_ROOT,
    rag_manager=rag_manager,
)
self_check_service = SelfCheckService(
    llm_manager=llm_manager,
    run_llm_call=run_llm_call,
)
routing_manager = MultiSignalRouter(
    run_llm_call=run_llm_call,
    tool_definitions=tool_service.function_definitions,
    embedder=shared_embedder,
)


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
    """GitHub 저장소를 클론하고 요약/청크/심볼링크 생성 작업을 수행"""
    repo_name = url.split("/")[-1].replace(".git", "")
    dest = GIT_CLONE_DIR / repo_name
    git_dir = dest / ".git"

    try:
        # ✅ 폴더 존재 + .git 폴더도 있으면 pull
        if dest.exists() and git_dir.exists():
            await asyncio.to_thread(subprocess.run, ["git", "-C", str(dest), "pull"], check=True)
        else:
            # ⚠️ 기존 폴더가 남아있고 .git이 없으면 제거 후 재clone
            if dest.exists():
                shutil.rmtree(dest)
            await asyncio.to_thread(subprocess.run, ["git", "clone", url, str(dest)], check=True)

        # ✅ DB 기록 및 분석 단계
        repo_id = await asyncio.to_thread(insert_repo_to_db, repo_name, url, dest)

        await broadcast({"type": "git_status", "text": "Summarizing files..."})
        await asyncio.to_thread(agent.summarize_repo_files, repo_id, dest)

        await broadcast({"type": "git_status", "text": "Generating chunks..."})
        await asyncio.to_thread(agent.chunk_repo_files, repo_id, dest)

        await broadcast({"type": "git_status", "text": "Extracting symbol links..."})
        await asyncio.to_thread(agent.extract_symbol_links, repo_id, dest)

        await broadcast({"type": "git_status", "text": "✅ Done."})

    except subprocess.CalledProcessError as e:
        # pull/clone 명령이 실패할 경우 재시도
        await broadcast({"type": "git_status", "text": f"⚠️ Git command failed: {e}. Retrying..."})
        if dest.exists():
            shutil.rmtree(dest)
        await asyncio.to_thread(subprocess.run, ["git", "clone", url, str(dest)], check=True)
        await broadcast({"type": "git_status", "text": "✅ Repository re-cloned successfully."})

    except Exception as e:
        await broadcast({"type": "error", "text": f"❌ Repository clone failed: {e}"})
        print(f"[Bridge] ⚠️ clone_repo_and_broadcast failed: {e}")


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
