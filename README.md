# CALL4 Case Clustering

This repository currently contains a local stdio MCP bridge for CALL4's public
flight API MCP server.

## CALL4 MCP stdio bridge

CALL4 exposes its MCP server as SSE + JSON-RPC. Codex's `mcp add --url` expects
a Streamable HTTP MCP endpoint, so the direct `/flight_api/mcp/sse` registration
does not reliably expose tools inside Codex.

The bridge in `mcp-bridge/call4-flight-stdio.mjs` presents a stdio MCP server to
Codex and proxies tool calls to CALL4.

### Tools

- `search_cases`
- `get_case_details`
- `list_documents`
- `fetch_document_text`
- `get_supporter_voices`
- `get_updates`
- `get_calendars`

### Registration

Register it with Codex as a local command MCP server:

```bash
/Applications/ChatGPT.app/Contents/Resources/codex mcp remove call4_flight_api
/Applications/ChatGPT.app/Contents/Resources/codex mcp add call4_flight_api -- node /Users/k.tokunaga/call4/case_clustering/mcp-bridge/call4-flight-stdio.mjs
```

The bridge uses `https://www.call4.jp/flight_api/mcp/sse` by default. Override it
with `CALL4_MCP_ENDPOINT` if needed.
