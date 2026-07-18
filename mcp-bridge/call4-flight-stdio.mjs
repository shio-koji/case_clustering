#!/usr/bin/env node

const REMOTE_ENDPOINT =
  process.env.CALL4_MCP_ENDPOINT ?? "https://www.call4.jp/flight_api/mcp/sse";

const TOOLS = [
  {
    name: "search_cases",
    description: "キーワードやタグでCALL4の公開ケースを検索します。",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "検索キーワード" },
        search_mode: {
          type: "string",
          enum: ["and", "or"],
          description: "検索モード（デフォルト: and）",
        },
        status: {
          type: "string",
          description:
            "互換性のため受け取りますが、取得対象は公開ケースに固定されます。",
        },
      },
    },
  },
  {
    name: "get_case_details",
    description: "特定のケースの詳細情報を取得します。",
    inputSchema: {
      type: "object",
      properties: { id: { type: "string", description: "ケースID" } },
      required: ["id"],
    },
  },
  {
    name: "list_documents",
    description: "ケースに関連する訴訟資料の一覧を取得します。",
    inputSchema: {
      type: "object",
      properties: { id: { type: "string", description: "ケースID" } },
      required: ["id"],
    },
  },
  {
    name: "fetch_document_text",
    description: "資料（PDF）から抽出されたテキスト内容を取得します。",
    inputSchema: {
      type: "object",
      properties: { id: { type: "string", description: "資料ID" } },
      required: ["id"],
    },
  },
  {
    name: "get_supporter_voices",
    description: "支援者からのコメントを取得します。",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "ケースID" },
        page: {
          type: "integer",
          description: "ページ番号（1始まり、デフォルト: 1）",
        },
        limit: {
          type: "integer",
          description: "1ページあたりの件数（デフォルト: 10、最大: 100）",
        },
      },
      required: ["id"],
    },
  },
  {
    name: "get_updates",
    description: "訴訟の進捗情報を取得します。",
    inputSchema: {
      type: "object",
      properties: { id: { type: "string", description: "ケースID" } },
      required: ["id"],
    },
  },
  {
    name: "get_calendars",
    description: "期日・裁判日程を取得します。",
    inputSchema: {
      type: "object",
      properties: { id: { type: "string", description: "ケースID" } },
      required: ["id"],
    },
  },
];

let inputBuffer = Buffer.alloc(0);
let nextRemoteId = 1;
let cookieHeader = "";
let remoteInitialized = false;

process.stdin.on("data", (chunk) => {
  inputBuffer = Buffer.concat([inputBuffer, chunk]);
  void drainInput();
});

process.stdin.resume();

async function drainInput() {
  while (true) {
    const headerEnd = inputBuffer.indexOf("\r\n\r\n");
    if (headerEnd === -1) return;

    const header = inputBuffer.subarray(0, headerEnd).toString("utf8");
    const match = header.match(/^Content-Length:\s*(\d+)/im);
    if (!match) {
      inputBuffer = inputBuffer.subarray(headerEnd + 4);
      continue;
    }

    const length = Number(match[1]);
    const messageStart = headerEnd + 4;
    const messageEnd = messageStart + length;
    if (inputBuffer.length < messageEnd) return;

    const rawMessage = inputBuffer.subarray(messageStart, messageEnd).toString("utf8");
    inputBuffer = inputBuffer.subarray(messageEnd);

    let message;
    try {
      message = JSON.parse(rawMessage);
    } catch (error) {
      sendError(null, -32700, `Parse error: ${error.message}`);
      continue;
    }

    handleMessage(message).catch((error) => {
      if (message.id !== undefined) {
        sendError(message.id, -32603, error.message);
      }
    });
  }
}

async function handleMessage(message) {
  if (message.id === undefined) {
    return;
  }

  switch (message.method) {
    case "initialize":
      sendResult(message.id, {
        protocolVersion: "2024-11-05",
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: "call4-flight-stdio-bridge", version: "0.1.0" },
      });
      break;

    case "ping":
      sendResult(message.id, {});
      break;

    case "tools/list":
      sendResult(message.id, { tools: TOOLS });
      break;

    case "tools/call":
      sendResult(message.id, await callRemoteTool(message.params));
      break;

    default:
      sendError(message.id, -32601, `Method not found: ${message.method}`);
  }
}

async function callRemoteTool(params = {}) {
  const name = params.name;
  const args = params.arguments ?? {};

  if (!TOOLS.some((tool) => tool.name === name)) {
    return {
      isError: true,
      content: [{ type: "text", text: `Unknown CALL4 tool: ${name}` }],
    };
  }

  try {
    await ensureRemoteInitialized();
    const response = await remoteRpc("tools/call", { name, arguments: args });
    return response.result;
  } catch (error) {
    return {
      isError: true,
      content: [{ type: "text", text: `CALL4 MCP bridge error: ${error.message}` }],
    };
  }
}

async function ensureRemoteInitialized() {
  if (remoteInitialized) return;

  await remoteRpc("initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "call4-flight-stdio-bridge", version: "0.1.0" },
  });

  await remoteRpc("notifications/initialized", {}, { notification: true });
  remoteInitialized = true;
}

async function remoteRpc(method, params, options = {}) {
  const payload = {
    jsonrpc: "2.0",
    method,
    params,
  };
  if (!options.notification) {
    payload.id = nextRemoteId++;
  }

  const response = await fetch(REMOTE_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
      ...(cookieHeader ? { Cookie: cookieHeader } : {}),
    },
    body: JSON.stringify(payload),
  });

  updateCookies(response);

  const body = await response.text();
  if (!response.ok) {
    throw new Error(`Remote HTTP ${response.status}: ${body.slice(0, 500)}`);
  }

  if (options.notification || body.trim() === "") {
    return {};
  }

  const json = parseRemoteJson(body);
  if (json.error) {
    throw new Error(json.error.message ?? JSON.stringify(json.error));
  }
  return json;
}

function parseRemoteJson(body) {
  const trimmed = body.trim();
  if (trimmed.startsWith("event:") || trimmed.startsWith("data:")) {
    const dataLine = trimmed
      .split(/\r?\n/)
      .find((line) => line.startsWith("data:"));
    if (!dataLine) {
      throw new Error(`Remote SSE response did not include data: ${trimmed}`);
    }
    return JSON.parse(dataLine.slice(5).trim());
  }
  return JSON.parse(trimmed);
}

function updateCookies(response) {
  const setCookie = response.headers.get("set-cookie");
  if (!setCookie) return;

  cookieHeader = setCookie
    .split(/,(?=[^;,]+=)/)
    .map((cookie) => cookie.split(";")[0].trim())
    .filter(Boolean)
    .join("; ");
}

function sendResult(id, result) {
  send({ jsonrpc: "2.0", id, result });
}

function sendError(id, code, message) {
  send({ jsonrpc: "2.0", id, error: { code, message } });
}

function send(message) {
  const body = JSON.stringify(message);
  const bytes = Buffer.byteLength(body, "utf8");
  process.stdout.write(`Content-Length: ${bytes}\r\n\r\n${body}`);
}
