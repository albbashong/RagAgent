import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

type Msg = {
  type: string;
  text?: string;
  data?: any;
  direction?: "sent" | "received";
  tabId?: number;
  timestamp?: string;
};

type TreeNode = {
  name: string;
  path?: string;
  type: "file" | "folder" | "error";
  children?: TreeNode[];
};

type PendingPrompt = {
  tabId?: number;
  text: string;
};

const formatTime = (ts?: string) => {
  if (!ts) return "";
  const date = new Date(ts);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString();
};

const normalizeMessage = (msg: Msg): Msg => ({
  ...msg,
  direction: msg.direction ?? "received",
  timestamp: msg.timestamp ?? new Date().toISOString(),
});

const buildMessageKey = (msg: Msg) =>
  `${msg.timestamp ?? ""}|${msg.type}|${
    msg.text ?? JSON.stringify(msg.data ?? {})
  }|${msg.tabId ?? ""}`;

const handleResetDB = async () => {
  try {
    const res = await axios.post("http://localhost:9013/reset_db");
    const data = res.data;
    if (data.status === "ok") {
      alert("✅ Database reset complete!");
    } else {
      alert(`⚠️ Reset failed: ${data.message}`);
    }
  } catch (err) {
    console.error("DB reset failed:", err);
    alert("❌ Error while resetting database.");
  }
};

// ✅ 폴더 트리 컴포넌트
const TreeView = ({
  node,
  onFileClick,
  onFolderExpand,
}: {
  node: TreeNode;
  onFileClick: (path: string) => void;
  onFolderExpand: (path: string) => void;
}) => {
  const [open, setOpen] = useState(false);
  const hasChildren = node.children && node.children.length > 0;

  const handleClick = async () => {
    if (node.type === "folder") {
      if (!open) {
        onFolderExpand(node.path ?? "");
      }
      setOpen((prev) => !prev);
    } else if (node.type === "file" && node.path) {
      onFileClick(node.path);
    }
  };

  return (
    <div className="ml-2">
      <div
        className={`cursor-pointer flex items-center gap-1 select-none ${
          node.type === "folder"
            ? "font-semibold text-blue-600"
            : "text-gray-700"
        }`}
        onClick={handleClick}
      >
        {node.type === "folder" ? (open ? "📂" : "📁") : "📄"} {node.name}
      </div>

      {hasChildren && open && (
        <div className="ml-4 border-l border-gray-300 pl-2">
          {node.children!.map((child, idx) => (
            <TreeView
              key={idx}
              node={child}
              onFileClick={onFileClick}
              onFolderExpand={onFolderExpand}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [connected, setConnected] = useState(false);
  const [dirTree, setDirTree] = useState<TreeNode | null>(null);
  const [activePanel, setActivePanel] = useState<"chat" | "file">("chat");
  const [activeConversation, setActiveConversation] = useState<"all" | number>(
    "all",
  );
  const [availableTabs, setAvailableTabs] = useState<number[]>([]);
  const [composeTarget, setComposeTarget] = useState<"auto" | number>("auto");
  const [fileContent, setFileContent] = useState("");
  const [mainInput, setMainInput] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState<PendingPrompt | null>(
    null,
  );
  const [pendingReply, setPendingReply] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const messageKeysRef = useRef<Set<string>>(new Set());
  const tabSetRef = useRef<Set<number>>(new Set());

  const pushMessages = (incoming: Msg[]) => {
    if (!incoming.length) return;

    const toAppend: Msg[] = [];
    let sawPending: PendingPrompt | null = null;
    let tabsChanged = false;

    incoming.forEach((raw) => {
      const normalized = normalizeMessage(raw);
      const key = buildMessageKey(normalized);
      if (messageKeysRef.current.has(key)) {
        return;
      }
      messageKeysRef.current.add(key);
      toAppend.push(normalized);

      if (
        typeof normalized.tabId === "number" &&
        !tabSetRef.current.has(normalized.tabId)
      ) {
        tabSetRef.current.add(normalized.tabId);
        tabsChanged = true;
      }

      if (normalized.type === "pending_request") {
        sawPending = {
          tabId: normalized.tabId,
          text:
            normalized.text ??
            JSON.stringify(normalized.data ?? "", null, 2),
        };
      }
    });

    if (toAppend.length) {
      setMessages((prev) => [...prev, ...toAppend]);
    }

    if (tabsChanged) {
      const sorted = Array.from(tabSetRef.current).sort((a, b) => a - b);
      setAvailableTabs(sorted);
      setActiveConversation((prev) =>
        prev === "all" && sorted.length ? sorted[sorted.length - 1] : prev,
      );
    }

    if (sawPending) {
      setPendingPrompt(sawPending);
      setActivePanel("chat");
      setActiveConversation((prev) =>
        typeof sawPending!.tabId === "number" ? sawPending!.tabId : prev,
      );
    }
  };

  // ✅ 메시지 스크롤 자동 이동
  const filteredMessages = useMemo(
    () =>
      activeConversation === "all"
        ? messages
        : messages.filter((msg) => msg.tabId === activeConversation),
    [messages, activeConversation],
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [filteredMessages]);

  // ✅ 앱 시작 시 ./workspace 트리 초기 불러오기
  useEffect(() => {
    const fetchInitialTree = async () => {
      try {
        const res = await axios.get("http://localhost:9013/init_tree");
        const data = res.data;
        console.log("🌳 초기 트리 응답:", data);

        if (data.status === "ok") {
          if (data.trees && Array.isArray(data.trees)) {
            setDirTree({
              name: "workspace",
              type: "folder",
              children: data.trees,
            });
          } else if (data.tree) {
            setDirTree(data.tree);
          }
        } else if (data.status === "empty") {
          setDirTree({
            name: "workspace",
            type: "folder",
            children: [{ name: "📁 (Empty directory)", type: "error" }],
          });
        } else {
          setDirTree({
            name: "workspace",
            type: "error",
            children: [{ name: `⚠️ ${data.message}`, type: "error" }],
          });
        }
      } catch (e) {
        console.error("초기 트리 불러오기 실패:", e);
        setDirTree({
          name: "workspace",
          type: "error",
          children: [{ name: "⚠️ Cannot fetch directory", type: "error" }],
        });
      }
    };

    fetchInitialTree();
  }, []);

  // ✅ 기존 메시지 히스토리 로드
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get("http://localhost:9013/history?limit=200");
        const data = res.data;
        if (data?.status === "ok" && Array.isArray(data.messages)) {
          pushMessages(data.messages as Msg[]);
        }
      } catch (err) {
        console.error("히스토리 불러오기 실패:", err);
      }
    };
    fetchHistory();
  }, []);

  // ✅ WebSocket 연결
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://localhost:9013/ws/client`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "dir_tree" && msg.data) {
          setDirTree(msg.data);
        }
        pushMessages([msg]);
      } catch (e) {
        console.error("메시지 파싱 실패:", e);
      }
    };

    return () => ws.close();
  }, []);

  // ✅ 특정 폴더 하위 트리 요청 및 병합
  const handleFolderExpand = async (path: string) => {
    if (!path) return;
    try {
      const res = await axios.get(
        `http://localhost:9013/tree?path=${encodeURIComponent(path)}`,
      );
      const data = res.data;
      if (data.status === "ok") {
        const subtree = data.tree.children;
        setDirTree((prev) =>
          prev ? updateTreeNode(prev, path, subtree) : prev,
        );
      }
    } catch (e) {
      console.error("하위 트리 불러오기 실패:", e);
    }
  };

  const updateTreeNode = (
    node: TreeNode,
    targetPath: string,
    newChildren: TreeNode[],
  ): TreeNode => {
    if (node.path === targetPath) {
      return { ...node, children: newChildren };
    }
    if (!node.children) return node;
    return {
      ...node,
      children: node.children.map((child) =>
        updateTreeNode(child, targetPath, newChildren),
      ),
    };
  };

  // ✅ 파일 클릭 시 FastAPI에서 내용 가져오기
  const handleFileClick = async (path: string) => {
    try {
      const res = await axios.get(
        `http://localhost:9013/file?path=${encodeURIComponent(path)}`,
      );
      const data = res.data;
      if (data.status === "ok") {
        setFileContent(data.content);
        setActivePanel("file");
      } else {
        setFileContent(`⚠️ ${data.message}`);
        setActivePanel("file");
      }
    } catch (e) {
      setFileContent("⚠️ 파일을 불러오는 중 오류가 발생했습니다.");
      setActivePanel("file");
    }
  };

  const sendByPost = async (payload: {
    type: string;
    text: string;
    tabId?: number;
  }) => {
    try {
      await axios.post("http://localhost:9013/send", payload);
    } catch (err) {
      console.error("POST /send 실패:", err);
      throw err;
    }
  };

  const sendMain = async () => {
    const text = mainInput.trim();
    if (!text) return;
    const resolved =
      composeTarget === "auto"
        ? activeConversation === "all"
          ? undefined
          : activeConversation
        : composeTarget;

    setMainInput("");
    try {
      await sendByPost({
        type: "user_input",
        text,
        tabId: typeof resolved === "number" ? resolved : undefined,
      });
    } catch (err) {
      // already logged
    }

    const localMsg: Msg = {
      type: "main_input(local)",
      text,
      direction: "sent",
      timestamp: new Date().toISOString(),
    };
    if (typeof resolved === "number") {
      localMsg.tabId = resolved;
    }
    pushMessages([localMsg]);
  };

  const sendPendingResponse = async (text: string, target: PendingPrompt) => {
    try {
      await sendByPost({
        type: "pending_response",
        text,
        tabId: target.tabId,
      });
    } catch (err) {
      // already logged
    }

    const localMsg: Msg = {
      type: "pending_response(local)",
      text,
      direction: "sent",
      timestamp: new Date().toISOString(),
    };
    if (typeof target.tabId === "number") {
      localMsg.tabId = target.tabId;
    }
    pushMessages([localMsg]);
  };

  const handlePendingSubmit = async () => {
    if (!pendingPrompt) return;
    const content = pendingReply.trim();
    if (!content) return;
    const currentPrompt = pendingPrompt;
    setPendingReply("");
    setPendingPrompt(null);
    await sendPendingResponse(content, currentPrompt);
  };

  const handlePendingQuick = async (preset: string) => {
    if (!pendingPrompt) return;
    const currentPrompt = pendingPrompt;
    setPendingReply("");
    setPendingPrompt(null);
    await sendPendingResponse(preset, currentPrompt);
  };

  const dismissPending = () => {
    setPendingPrompt(null);
    setPendingReply("");
  };

  const handleReset = async () => {
    try {
      await sendByPost({ type: "reset", text: "" });
    } catch (err) {
      // already logged
    }
    pushMessages([
      {
        type: "reset(local)",
        text: "Reset request sent to supervisor.",
        direction: "sent",
        timestamp: new Date().toISOString(),
      },
    ]);
  };

  return (
    <div className="h-screen w-screen flex bg-gradient-to-br from-gray-100 to-gray-200">
      <div className="flex flex-col flex-1 h-full bg-white shadow-2xl border-r border-gray-200 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b bg-gradient-to-r from-sky-400 to-sky-600 text-white shadow-md">
          <h1 className="text-xl font-semibold">Supervisor Bridge UI</h1>
          <div className="flex items-center gap-3">
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium shadow ${
                connected ? "bg-green-500" : "bg-red-400"
              }`}
            >
              {connected ? "Connected" : "Disconnected"}
            </span>
            <button
              onClick={handleReset}
              className="text-sm px-3 py-1 rounded-full bg-white/20 hover:bg-white/30 transition"
            >
              Reset LLM
            </button>
            <button
              onClick={handleResetDB}
              className="text-sm px-3 py-1 rounded-full bg-red-500/80 hover:bg-red-600 text-white shadow transition"
            >
              Reset DB
            </button>            
          </div>
        </div>

        <div className="flex border-b bg-gray-50">
          <button
            onClick={() => setActivePanel("chat")}
            className={`flex-1 py-2 ${
              activePanel === "chat"
                ? "border-b-2 border-sky-500 text-sky-600 font-semibold"
                : "text-gray-500"
            }`}
          >
            🗨 Chat
          </button>
          <button
            onClick={() => setActivePanel("file")}
            className={`flex-1 py-2 ${
              activePanel === "file"
                ? "border-b-2 border-sky-500 text-sky-600 font-semibold"
                : "text-gray-500"
            }`}
          >
            📄 File Viewer
          </button>
        </div>

        <div className="flex-1 overflow-hidden bg-gray-50">
          {activePanel === "chat" ? (
            <div className="h-full flex flex-col p-4">
              <div className="flex flex-wrap gap-2 pb-3 border-b border-slate-200">
                <button
                  onClick={() => setActiveConversation("all")}
                  className={`px-3 py-1 rounded-full text-sm ${
                    activeConversation === "all"
                      ? "bg-sky-500 text-white shadow"
                      : "bg-white text-gray-600 border border-sky-200"
                  }`}
                >
                  All Tabs
                </button>
                {availableTabs.map((tabId) => (
                  <button
                    key={tabId}
                    onClick={() => setActiveConversation(tabId)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      activeConversation === tabId
                        ? "bg-sky-500 text-white shadow"
                        : "bg-white text-gray-600 border border-sky-200"
                    }`}
                  >
                    Tab #{tabId}
                  </button>
                ))}
              </div>

              <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto scroll-smooth py-4 space-y-3"
              >
                {filteredMessages.length === 0 ? (
                  <div className="text-sm text-slate-500">
                    아직 메시지가 없습니다.
                  </div>
                ) : (
                  filteredMessages.map((m, i) => {
                    const timeLabel = formatTime(m.timestamp);
                    return (
                      <div
                        key={`${i}-${m.timestamp}`}
                        className={`p-3 rounded-xl shadow-sm max-w-[75%] ${
                          m.direction === "sent"
                            ? "bg-gradient-to-r from-amber-200 to-amber-300 text-amber-900 ml-auto"
                            : "bg-amber-100 text-amber-800"
                        }`}
                      >
                        <div className="text-xs opacity-70 mb-1 flex items-center justify-between gap-3">
                          <span>{m.type}</span>
                          <span className="flex items-center gap-2">
                            {typeof m.tabId === "number" && (
                              <span className="px-2 py-0.5 rounded-full bg-white/40 text-xs">
                                Tab #{m.tabId}
                              </span>
                            )}
                            {timeLabel && <span>{timeLabel}</span>}
                          </span>
                        </div>
                        <pre className="text-sm whitespace-pre-wrap break-words">
                          {m.text ?? JSON.stringify(m.data ?? m, null, 2)}
                        </pre>
                      </div>
                    );
                  })
                )}
              </div>

              {pendingPrompt && (
                <div className="mt-3 border border-amber-300 bg-amber-50 rounded-xl p-4 shadow-inner">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-amber-700 flex items-center gap-2">
                      Pending Request
                      {typeof pendingPrompt.tabId === "number" && (
                        <span className="px-2 py-0.5 rounded-full bg-amber-200 text-xs text-amber-900">
                          Tab #{pendingPrompt.tabId}
                        </span>
                      )}
                    </h3>
                    <button
                      onClick={dismissPending}
                      className="text-xs text-amber-700 hover:underline"
                    >
                      Dismiss
                    </button>
                  </div>
                  <p className="text-sm text-amber-800 whitespace-pre-wrap mb-3">
                    {pendingPrompt.text}
                  </p>
                  <textarea
                    className="w-full border border-amber-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                    rows={3}
                    value={pendingReply}
                    onChange={(e) => setPendingReply(e.target.value)}
                    placeholder="Reply to the pending request..."
                  />
                  <div className="flex flex-wrap gap-2 justify-between mt-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => handlePendingQuick("Yes")}
                        className="px-3 py-1 rounded-full bg-amber-500 text-white text-sm shadow hover:opacity-90 transition"
                      >
                        ✅ Approve
                      </button>
                      <button
                        onClick={() => handlePendingQuick("No")}
                        className="px-3 py-1 rounded-full bg-red-500 text-white text-sm shadow hover:opacity-90 transition"
                      >
                        ❌ Decline
                      </button>
                      <button
                        onClick={() => handlePendingQuick("Revise")}
                        className="px-3 py-1 rounded-full bg-blue-500 text-white text-sm shadow hover:opacity-90 transition"
                      >
                        ✏️ Revise
                      </button>
                    </div>
                    <button
                      onClick={handlePendingSubmit}
                      className="px-4 py-1.5 rounded-full bg-amber-600 text-white text-sm shadow hover:opacity-90 transition"
                    >
                      Send Response
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <pre className="p-4 whitespace-pre-wrap break-words text-sm font-mono bg-gray-900 text-gray-100 h-full overflow-auto">
              {fileContent || "📁 파일을 선택하세요."}
            </pre>
          )}
        </div>

        {activePanel === "chat" && (
          <div className="p-3 border-t bg-white flex flex-wrap gap-2 items-center">
            <input
              className="flex-1 min-w-[12rem] border rounded-full px-4 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
              value={mainInput}
              onChange={(e) => setMainInput(e.target.value)}
              placeholder="Type your message..."
              onKeyDown={(e) => e.key === "Enter" && sendMain()}
            />
            <select
              value={composeTarget === "auto" ? "auto" : String(composeTarget)}
              onChange={(e) =>
                setComposeTarget(
                  e.target.value === "auto" ? "auto" : Number(e.target.value),
                )
              }
              className="border rounded-full px-3 py-2 bg-white shadow-sm text-sm"
            >
              <option value="auto">Auto Tab</option>
              {availableTabs.map((tabId) => (
                <option key={tabId} value={tabId}>
                  Tab #{tabId}
                </option>
              ))}
            </select>
            <button
              onClick={sendMain}
              className="px-5 py-2 rounded-full bg-gradient-to-r from-amber-500 to-orange-400 text-white font-semibold shadow hover:opacity-90 transition"
            >
              Send
            </button>
          </div>
        )}
      </div>

      {/* 오른쪽: 폴더 트리 */}
      <div className="w-[28rem] bg-white overflow-y-auto border-l border-gray-300 shadow-inner p-4">
        <h2 className="text-lg font-semibold text-gray-800 mb-3 border-b pb-2">
          📁 Repository Structure
        </h2>
        {dirTree ? (
          <TreeView
            node={dirTree}
            onFileClick={handleFileClick}
            onFolderExpand={handleFolderExpand}
          />
        ) : (
          <div className="text-gray-400 italic mt-4">
            No directory loaded yet
          </div>
        )}
      </div>
    </div>
  );
}