import { useEffect, useRef, useState, useCallback } from "react";

type Msg = {
  type: string;
  text?: string;
  data?: any;
  direction: "sent" | "received"; // ✅ 보낸/받은 구분
  tabId?: number;
};

const Tab = ({ tabTitle, isSelectedTab, onClick, onClose, tabsCount }) => (
  <div
    className={`flex items-center px-3 py-1.5 cursor-pointer text-sm font-medium rounded-full transition-colors duration-200 shadow-sm
      ${
        isSelectedTab
          ? "bg-gradient-to-r from-teal-500 to-teal-400 text-white"
          : "bg-gray-200 text-gray-700 hover:bg-gray-300"
      }`}
    onClick={onClick}
  >
    <span>{tabTitle}</span>
    {tabsCount > 1 && (
      <button
        onClick={onClose}
        className="ml-2 w-4 h-4 flex items-center justify-center text-xs font-bold text-white bg-gray-500 rounded-full hover:bg-gray-600 transition"
      >
        &times;
      </button>
    )}
  </div>
);

export default function App() {
  const [tabs, setTabs] = useState<{ id: number; title: string }[]>([
    { id: 1, title: "Action #1" },
  ]);
  const [activeTabId, setActiveTabId] = useState(1);

  const [messages, setMessages] = useState<{ [key: number]: Msg[] }>({ 1: [] });
  const [rightMessages, setRightMessages] = useState<Msg[]>([]);

  const [connected, setConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const activeTabIdRef = useRef(activeTabId);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const [mainInput, setMainInput] = useState("");
  const [sessionInputs, setSessionInputs] = useState<Record<number, string>>(
    {}
  );

  useEffect(() => {
    activeTabIdRef.current = activeTabId;
  }, [activeTabId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages[activeTabId]]);

  const addLeftMessage = useCallback((msg: Msg, targetTabId?: number) => {
    setMessages((prev) => {
      const tabId = targetTabId || activeTabIdRef.current;
      const updated = { ...prev };
      if (!updated[tabId]) updated[tabId] = [];
      updated[tabId] = [...updated[tabId], msg];
      return updated;
    });
  }, []);

  const addRightMessage = useCallback((msg: Msg) => {
    setRightMessages((prev) => [...prev, msg]);
  }, []);

  // ✅ WebSocket 연결 (Supervisor → React 수신)
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://localhost:9013/ws/client`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (ev) => {
    try {
      const msg: Msg = JSON.parse(ev.data);

      switch (msg.type) {
        case "pending_request":
        case "session_input":
          if (msg.tabId) {
            addLeftMessage({ ...msg, direction: "received" }, msg.tabId);
          } else {
            console.warn("pending_request but no tabId:", msg);
          }
          break;

        case "main_input":
        case "main_log":
          addRightMessage({ ...msg, direction: "received" });
          break;

        default:
          // 기타 메시지 (system, error 등)
          addRightMessage({ ...msg, direction: "received" });
          break;
      }
    } catch (e) {
      console.error("메시지 파싱 실패:", e);
    }
  }

    return () => ws.close();
  }, [addLeftMessage, addRightMessage]);

  // ✅ POST 방식으로 메시지 보내기
  const sendByPost = async (payload: {
    type: string;
    text: string;
    tabId?: number;
  }) => {
    try {
      const res = await fetch("http://localhost:9013/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await res.json().catch(() => ({}));
    } catch (err) {
      console.error("POST /send 실패:", err);
    }
  };

  // --- 메인 입력 보내기 (POST 사용)
  const sendMain = async () => {
    const text = mainInput.trim();
    if (!text) return;
    setMainInput("");

    // Supervisor로 POST
    await sendByPost({ type: "user_input", text });

    // 로컬에도 반영
    addRightMessage({ type: "main_input(local)", text, direction: "sent" });

    if (text.toLowerCase() === "action") {
      const newTabId =
        tabs.length > 0 ? Math.max(...tabs.map((t) => t.id)) + 1 : 1;
      const newTab = { id: newTabId, title: `Action #${newTabId}` };
      setTabs((prev) => [...prev, newTab]);
      setMessages((prev) => ({
        ...prev,
        [newTabId]: [
          {
            type: "system",
            text: "New action tab created!",
            direction: "received",
          },
        ],
      }));
      setActiveTabId(newTabId);
      addLeftMessage(
        { type: "action(local)", text: "Action started", direction: "sent" },
        newTabId
      );
    }
  };

  // --- 세션 입력 보내기 (POST 사용)
  const sendToSession = async (tabId: number) => {
    const text = sessionInputs[tabId]?.trim();
    if (!text) return;
    setSessionInputs((prev) => ({ ...prev, [tabId]: "" }));

    // Supervisor로 POST
    await sendByPost({ type: "user_input_pending", text, tabId });

    // 로컬에도 반영
    addLeftMessage(
      { type: "session_input(local)", text, direction: "sent" },
      tabId
    );
  };

  // 탭 닫기
  const handleCloseTab = (tabIdToClose: number) => {
    setTabs((prevTabs) => {
      const newTabs = prevTabs.filter((tab) => tab.id !== tabIdToClose);
      if (newTabs.length === 0) {
        const newTab = { id: 1, title: "Action #1" };
        setMessages({
          1: [
            { type: "system", text: "New tab created!", direction: "received" },
          ],
        });
        setActiveTabId(1);
        return [newTab];
      }
      if (activeTabId === tabIdToClose) {
        const nextActive =
          newTabs[
            newTabs.findIndex((t) => t.id > tabIdToClose) % newTabs.length
          ] || newTabs[0];
        setActiveTabId(nextActive.id);
      }
      return newTabs;
    });

    setMessages((prev) => {
      const { [tabIdToClose]: _, ...rest } = prev;
      return rest;
    });
  };

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200">
      <div className="w-[90%] h-[90%] bg-white rounded-2xl shadow-2xl flex flex-col border border-gray-100 overflow-hidden">
        {/* 헤더 */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b 
                bg-gradient-to-r from-sky-400 to-sky-600 
                text-white shadow-md"
        >
          <h1 className="text-xl font-semibold">Supervisor Bridge UI</h1>
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium shadow ${
              connected ? "bg-green-500 text-white" : "bg-red-400 text-white"
            }`}
          >
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>

        {/* 상단: Action Tabs | Main 영역 */}
        <div className="flex bg-gray-50 border-b">
          {/* 왼쪽: Action Tabs */}
          <div className="flex items-center space-x-2 px-4 py-2 w-2/3">
            {tabs.map((tab) => (
              <Tab
                key={tab.id}
                tabTitle={tab.title}
                isSelectedTab={tab.id === activeTabId}
                onClick={() => setActiveTabId(tab.id)}
                onClose={() => handleCloseTab(tab.id)}
                tabsCount={tabs.length}
              />
            ))}
          </div>

          {/* 세로 구분선 */}
          <div className="w-px bg-gray-200"></div>

          {/* 오른쪽: Main Message */}
          <div className="flex items-center px-4 py-2 w-1/3">
            <span
              className="px-3 py-1.5 text-sm font-medium rounded-full 
                        bg-gradient-to-r from-amber-500 to-orange-400 text-white shadow-sm"
            >
              Main Message
            </span>
          </div>
        </div>

        {/* 본문 */}
        <div className="flex-1 flex min-h-0">
          {/* 왼쪽: 세션 로그 + 입력창 */}
          <div className="w-2/3 flex flex-col bg-gray-50 min-h-0">
            <div className="flex-1 p-4 space-y-3 overflow-y-auto min-h-0">
              {messages[activeTabId]?.map((m, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-xl shadow-sm max-w-[75%] ${
                    m.direction === "sent"
                      ? "bg-gradient-to-r from-teal-100 to-teal-200 text-teal-900 ml-auto"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  <div className="text-xs opacity-60 mb-1">{m.type}</div>
                  <pre className="text-sm whitespace-pre-wrap break-words">
                    {m.text ?? JSON.stringify(m.data ?? m)}
                  </pre>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
            {/* 세션 입력창 */}
            <div className="p-3 border-t bg-white flex gap-2">
              <input
                className="flex-1 border rounded-full px-4 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-400"
                value={sessionInputs[activeTabId] || ""}
                onChange={(e) =>
                  setSessionInputs((prev) => ({
                    ...prev,
                    [activeTabId]: e.target.value,
                  }))
                }
                placeholder={`Reply to ${
                  tabs.find((t) => t.id === activeTabId)?.title
                }`}
                onKeyDown={(e) =>
                  e.key === "Enter" && sendToSession(activeTabId)
                }
              />
              <button
                onClick={() => sendToSession(activeTabId)}
                className="px-5 py-2 rounded-full bg-gradient-to-r from-teal-500 to-emerald-400 text-white font-semibold shadow hover:opacity-90 transition"
              >
                Send
              </button>
            </div>
          </div>

          <div className="w-px bg-gray-200"></div>

          {/* 오른쪽: 메인 로그 + 메인 입력창 */}
          <div className="w-1/3 flex flex-col bg-gray-50 overflow-y-auto min-h-0">
            <div className="flex-1 p-4 space-y-3 overflow-y-auto min-h-0">
              {rightMessages.map((m, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-xl shadow-sm max-w-[75%] ${
                    m.direction === "sent"
                      ? "bg-gradient-to-r from-amber-200 to-amber-300 text-amber-900 ml-auto"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  <div className="text-xs opacity-60 mb-1">{m.type}</div>
                  <pre className="text-sm whitespace-pre-wrap break-words">
                    {m.text}
                  </pre>
                </div>
              ))}
            </div>
            {/* 메인 입력창 */}
            <div className="p-3 border-t bg-white flex gap-2">
              <input
                className="flex-1 border rounded-full px-4 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                value={mainInput}
                onChange={(e) => setMainInput(e.target.value)}
                placeholder="Type main input…"
                onKeyDown={(e) => e.key === "Enter" && sendMain()}
              />
              <button
                onClick={sendMain}
                className="px-5 py-2 rounded-full bg-gradient-to-r from-amber-500 to-orange-400 text-white font-semibold shadow hover:opacity-90 transition"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
