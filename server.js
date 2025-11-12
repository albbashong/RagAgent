console.log("✅ server.js 시작됨");
const express = require("express");
const path = require("path");
const { createProxyMiddleware } = require("http-proxy-middleware");

const app = express();
const PORT = 9012; // React 웹 포트

// ✅ 1) /api 요청을 FastAPI(9013)로 프록시
app.use(
  "/api",
  createProxyMiddleware({
	  target: "http://localhost:9013", // FastAPI 서버 주소
    changeOrigin: true,
    logLevel: "debug", // 프록시 로그 확인
     pathRewrite: { "^/api": "" },
    onProxyReq: (proxyReq, req, res) => {
      console.log("🔥 프록시 잡힘:", req.method, req.url);
    },
    onError: (err, req, res) => {
      console.error("❌ 프록시 에러:", err);
      res.status(500).send("Proxy error");
    }
  })
);

// ✅ 2) React 정적 파일
app.use(express.static(path.join(__dirname, "dist")));

// ✅ 3) SPA fallback (단, /api/* 는 프록시가 먼저 처리해야 함)
// 모든 GET 요청 → React index.html
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

app.listen(PORT, () => {
  console.log(`✅ React app running at http://localhost:${PORT}`);
});

