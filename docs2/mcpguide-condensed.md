# MCP Server Development Reference

MCP (Model Context Protocol) is a JSON-RPC 2.0 standard for connecting AI models to external tools/data. Architecture: Host → Client → Server.

## Core Primitives

| Primitive | Purpose | Analogy |
|-----------|---------|---------|
| **Tool** | Executable function | POST endpoint |
| **Resource** | Read-only data | GET endpoint |
| **Prompt** | Reusable template | Stored procedure |

## Minimal Server (TypeScript)

```bash
npm install @modelcontextprotocol/sdk zod
```

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "my-server", version: "1.0.0" });

// Tool
server.tool("search", "Search database", {
  query: z.string(),
  limit: z.number().optional()
}, async ({ query, limit = 10 }) => ({
  content: [{ type: "text", text: JSON.stringify({ query, limit }) }]
}));

// Resource
server.resource("config://settings", "App settings", async () => ({
  contents: [{ uri: "config://settings", mimeType: "application/json", text: "{}" }]
}));

// Prompt
server.prompt("analyze", "Analyze code", {
  code: z.string()
}, async ({ code }) => ({
  messages: [{ role: "user", content: { type: "text", text: `Analyze:\n${code}` } }]
}));

await server.connect(new StdioServerTransport());
```

## Minimal Server (Python)

```bash
pip install fastmcp
```

```python
from fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def search(query: str, limit: int = 10) -> dict:
    """Search database."""
    return {"query": query, "limit": limit}

@mcp.resource("config://settings")
def get_settings() -> str:
    """App settings."""
    return "{}"

@mcp.prompt()
def analyze(code: str) -> str:
    """Analyze code."""
    return f"Analyze:\n{code}"

if __name__ == "__main__":
    mcp.run()
```

## Transport

| Type | Use | Config |
|------|-----|--------|
| stdio | Local/CLI | `StdioServerTransport()` |
| HTTP | Remote/Cloud | `StreamableHttpServerTransport()` |

**Critical:** Never `console.log()` in stdio servers—use `console.error()` for logging.

## Client Configuration

**Location:**
- Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
- Claude Code: `claude mcp add <name> -- <command>`
- VS Code: `.vscode/settings.json` → `mcp.servers`
- Cursor: `~/.cursor/mcp.json`

**Format:**
```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/build/index.js"],
      "env": { "API_KEY": "xxx" }
    }
  }
}
```

**Claude Code CLI:**
```bash
claude mcp add my-server -- node /path/server.js
claude mcp add -e API_KEY=xxx my-server -- npx -y @org/server
claude mcp list
claude mcp remove my-server
```

## Testing

```bash
npx @modelcontextprotocol/inspector node build/server.js
npx @modelcontextprotocol/inspector python server.py
```

Opens at `http://localhost:6274` with Tools/Resources/Prompts panels.

## Error Handling

```typescript
server.tool("risky", "May fail", { input: z.string() }, async ({ input }) => {
  try {
    return { content: [{ type: "text", text: await doWork(input) }] };
  } catch (e) {
    console.error(e); // Log to stderr
    return {
      content: [{ type: "text", text: `Failed: ${e.message}. Try X instead.` }],
      isError: true
    };
  }
});
```

## Security Essentials

- Validate all inputs with Zod schemas
- Use `process.env` for secrets, never hardcode
- Whitelist allowed values: `z.enum(["allowed", "values"])`
- Log errors server-side, return helpful (not sensitive) messages to AI

## HTTP Server (Remote)

```typescript
import express from "express";
import { StreamableHttpServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const app = express();
app.use(express.json());

app.all("/mcp", async (req, res) => {
  const transport = new StreamableHttpServerTransport();
  await server.connect(transport);
  await transport.handleRequest(req, res);
});

app.listen(3000);
```

## Protocol Reference

**Initialization:**
```
Client → initialize → Server
Server → response (capabilities) → Client
Client → initialized → Server
```

**Methods:**
| Method | Description |
|--------|-------------|
| `tools/list` | List available tools |
| `tools/call` | Execute tool |
| `resources/list` | List resources |
| `resources/read` | Read resource |
| `prompts/list` | List prompts |
| `prompts/get` | Get prompt |

**Tool Response:**
```typescript
{
  content: [
    { type: "text", text: "..." },
    { type: "image", data: "base64...", mimeType: "image/png" }
  ],
  isError?: boolean
}
```

## Links

- Spec: https://modelcontextprotocol.io/specification
- TS SDK: https://github.com/modelcontextprotocol/typescript-sdk
- Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Servers: https://github.com/modelcontextprotocol/servers
- Inspector: https://github.com/modelcontextprotocol/inspector
