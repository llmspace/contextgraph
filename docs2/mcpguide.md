# Complete MCP (Model Context Protocol) Developer Guide

A comprehensive guide to understanding, building, deploying, and integrating MCP servers with any application.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
4. [Transport Mechanisms](#transport-mechanisms)
5. [Building an MCP Server](#building-an-mcp-server)
6. [Testing and Debugging](#testing-and-debugging)
7. [Client Configuration](#client-configuration)
8. [Security Best Practices](#security-best-practices)
9. [Error Handling](#error-handling)
10. [Deployment](#deployment)
11. [Resources](#resources)

---

## Introduction

### What is MCP?

The **Model Context Protocol (MCP)** is an open standard introduced by Anthropic in November 2024 that standardizes how AI systems (like LLMs) integrate with external tools, databases, and data sources. Think of it as "the USB-C port for AI" - providing a uniform interface for connecting language models to external capabilities.

MCP enables developers to build **secure, two-way connections** between data sources and AI-powered tools through a standardized protocol based on JSON-RPC 2.0.

### Why MCP?

Before MCP, connecting LLMs to external tools required custom integrations for each combination of AI model and data source. MCP solves this by providing:

- **Standardization**: One protocol for all AI-to-tool connections
- **Security**: Built-in authentication and authorization patterns
- **Flexibility**: Support for local and remote servers
- **Ecosystem**: Growing library of pre-built servers and clients

### Key Adopters

MCP has been adopted by major AI providers including:
- **Anthropic** (Claude Desktop, Claude Code)
- **OpenAI**
- **Google DeepMind**
- **Microsoft** (VS Code Copilot)
- Development tools: Zed, Replit, Codeium, Cursor, Windsurf

---

## Architecture Overview

### Components

MCP uses a **client-server architecture** with three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                         MCP HOST                            │
│  (Claude Desktop, IDE, Custom Application)                  │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ MCP Client  │  │ MCP Client  │  │ MCP Client  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │MCP Server │   │MCP Server │   │MCP Server │
    │(Database) │   │(API)      │   │(Files)    │
    └───────────┘   └───────────┘   └───────────┘
```

1. **MCP Host**: The runtime environment (Claude Desktop, IDE, or custom app) that manages clients
2. **MCP Client**: Connects to servers, handles protocol communication
3. **MCP Server**: Exposes tools, resources, and prompts to AI models

### Communication Flow

```
Client                                    Server
  │                                          │
  │──────── initialize request ─────────────>│
  │<─────── initialize response ─────────────│
  │                                          │
  │──────── initialized notification ───────>│
  │                                          │
  │<══════ Normal Operations Begin ═════════>│
  │                                          │
  │──────── tools/list ─────────────────────>│
  │<─────── tool definitions ────────────────│
  │                                          │
  │──────── tools/call ─────────────────────>│
  │<─────── tool result ─────────────────────│
```

---

## Core Concepts

### The Three Primitives

MCP provides three core primitives for AI interaction:

#### 1. Tools (Executable Functions)

Tools are functions that LLMs can invoke to perform actions or retrieve information. They're like POST endpoints in a REST API.

```typescript
// Example tool definition
{
  name: "search_database",
  description: "Search the database for records matching a query",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "Search query" },
      limit: { type: "number", description: "Max results", default: 10 }
    },
    required: ["query"]
  }
}
```

**Use cases:**
- Execute code
- Query databases
- Call external APIs
- Perform calculations
- Manage files

#### 2. Resources (Read-Only Data)

Resources are data entities exposed by the server. They're like GET endpoints - providing context without side effects.

```typescript
// Example resource
{
  uri: "file:///config/settings.json",
  name: "Application Settings",
  description: "Current application configuration",
  mimeType: "application/json"
}
```

**Use cases:**
- File contents
- Database schemas
- Configuration data
- Static information

#### 3. Prompts (Reusable Templates)

Prompts are structured message templates that guide AI interactions. They define consistent patterns for common tasks.

```typescript
// Example prompt
{
  name: "code-review",
  description: "Review code for quality and best practices",
  arguments: [
    {
      name: "code",
      description: "The code to review",
      required: true
    },
    {
      name: "language",
      description: "Programming language",
      required: false
    }
  ]
}
```

**Use cases:**
- Standardized analysis workflows
- Guided interactions
- Consistent output formats

### JSON-RPC 2.0 Foundation

All MCP communication uses JSON-RPC 2.0. There are three message types:

#### Requests
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_database",
    "arguments": { "query": "users" }
  }
}
```

#### Responses
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      { "type": "text", "text": "Found 42 users" }
    ]
  }
}
```

#### Notifications (No Response Expected)
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "task-1",
    "progress": 50,
    "total": 100
  }
}
```

---

## Transport Mechanisms

MCP supports multiple transport options for different use cases:

### 1. stdio (Standard Input/Output)

**Best for:** Local tools, CLI integrations, development

The server runs as a subprocess, communicating via stdin/stdout.

```
┌────────────┐     stdin      ┌────────────┐
│   Client   │───────────────>│   Server   │
│            │<───────────────│  (Process) │
└────────────┘     stdout     └────────────┘
```

**Advantages:**
- Simple setup
- No network configuration
- Low overhead
- Best performance for single-client scenarios

**Important:** Never write to stdout except for JSON-RPC messages. Use stderr for logging:

```python
# WRONG - breaks the protocol
print("Debug message")

# CORRECT - use logging to stderr
import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("Debug message")
```

### 2. Streamable HTTP (Recommended for Remote)

**Best for:** Cloud deployments, multi-client scenarios, web applications

Introduced in MCP specification 2025-03-26, this is the modern standard for remote servers.

```
┌────────────┐    HTTP POST    ┌────────────┐
│   Client   │────────────────>│   Server   │
│            │<────────────────│  (Remote)  │
└────────────┘   Response/SSE  └────────────┘
```

**Features:**
- Single endpoint architecture
- Supports both request-response and streaming
- Compatible with serverless platforms
- Built-in session management via `Mcp-Session-Id` header

### 3. SSE (Server-Sent Events) - Deprecated

**Status:** Deprecated as of March 2025

The legacy HTTP+SSE transport required two endpoints and has been replaced by Streamable HTTP. Still supported for backward compatibility.

### Transport Comparison

| Feature | stdio | Streamable HTTP | SSE (Legacy) |
|---------|-------|-----------------|--------------|
| Use Case | Local | Remote | Remote |
| Multi-client | No | Yes | Yes |
| Serverless | N/A | Yes | No |
| Complexity | Low | Medium | High |
| Status | Active | Recommended | Deprecated |

---

## Building an MCP Server

### TypeScript Implementation

#### Installation

```bash
npm install @modelcontextprotocol/sdk
```

#### Basic stdio Server

```typescript
// server.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// Create the server
const server = new McpServer({
  name: "my-mcp-server",
  version: "1.0.0",
});

// Define a tool
server.tool(
  "add_numbers",
  "Add two numbers together",
  {
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  },
  async ({ a, b }) => {
    return {
      content: [
        {
          type: "text",
          text: `Result: ${a + b}`,
        },
      ],
    };
  }
);

// Define a resource
server.resource(
  "config://app/settings",
  "Application settings",
  async () => {
    return {
      contents: [
        {
          uri: "config://app/settings",
          mimeType: "application/json",
          text: JSON.stringify({ theme: "dark", language: "en" }),
        },
      ],
    };
  }
);

// Define a prompt
server.prompt(
  "summarize",
  "Summarize the given text",
  {
    text: z.string().describe("Text to summarize"),
    style: z.enum(["brief", "detailed"]).optional().describe("Summary style"),
  },
  async ({ text, style = "brief" }) => {
    return {
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Please provide a ${style} summary of:\n\n${text}`,
          },
        },
      ],
    };
  }
);

// Start the server with stdio transport
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP Server running on stdio"); // Note: stderr for logging
}

main().catch(console.error);
```

#### Build and Run

```bash
# Build
npx tsc

# Run directly
node build/server.js

# Or use with npx for development
npx tsx server.ts
```

### Python Implementation

#### Installation

```bash
pip install fastmcp
# or
pip install mcp
```

#### Basic Server with FastMCP

```python
# server.py
from fastmcp import FastMCP
import logging

# Configure logging to stderr
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the server
mcp = FastMCP("my-mcp-server")

# Define a tool using decorator
@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    logger.debug(f"Adding {a} + {b}")
    return a + b

@mcp.tool()
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for matching records.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching records
    """
    # Simulated database search
    return [
        {"id": 1, "name": "Result 1"},
        {"id": 2, "name": "Result 2"},
    ][:limit]

# Define a resource
@mcp.resource("config://settings")
def get_settings() -> str:
    """Get application settings."""
    import json
    return json.dumps({"theme": "dark", "version": "1.0"})

# Define a prompt
@mcp.prompt()
def analyze_code(code: str, language: str = "python") -> str:
    """Create a code analysis prompt.

    Args:
        code: The code to analyze
        language: Programming language
    """
    return f"""Please analyze the following {language} code for:
1. Potential bugs
2. Performance issues
3. Best practice violations

Code:
```{language}
{code}
```"""

# Run the server
if __name__ == "__main__":
    mcp.run()
```

#### Advanced Python Server with Official SDK

```python
# advanced_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    Prompt,
    PromptMessage,
    GetPromptResult,
)
import asyncio
import json

# Create server instance
server = Server("advanced-mcp-server")

# Tool handlers
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_database",
            description="Execute a database query",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query"},
                    "database": {"type": "string", "description": "Database name"},
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="send_email",
            description="Send an email message",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_database":
        # Simulated database query
        results = [{"id": 1, "value": "example"}]
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    elif name == "send_email":
        # Simulated email sending
        return [TextContent(
            type="text",
            text=f"Email sent to {arguments['to']}"
        )]

    raise ValueError(f"Unknown tool: {name}")

# Resource handlers
@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="file:///data/schema.sql",
            name="Database Schema",
            description="Current database schema definition",
            mimeType="application/sql",
        ),
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "file:///data/schema.sql":
        return "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));"
    raise ValueError(f"Unknown resource: {uri}")

# Prompt handlers
@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="debug-error",
            description="Help debug an error message",
            arguments=[
                {"name": "error", "description": "Error message", "required": True},
                {"name": "context", "description": "Additional context", "required": False},
            ],
        ),
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    if name == "debug-error":
        error = arguments.get("error", "")
        context = arguments.get("context", "No additional context")
        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Please help debug this error:\n\n{error}\n\nContext: {context}",
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown prompt: {name}")

# Main entry point
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Streamable HTTP Server (TypeScript)

```typescript
// http-server.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHttpServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { z } from "zod";

const app = express();
app.use(express.json());

const server = new McpServer({
  name: "http-mcp-server",
  version: "1.0.0",
});

// Define tools
server.tool(
  "fetch_weather",
  "Get current weather for a location",
  {
    location: z.string().describe("City name"),
  },
  async ({ location }) => {
    // Simulated weather fetch
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            location,
            temperature: 72,
            conditions: "Sunny",
          }),
        },
      ],
    };
  }
);

// Store sessions
const sessions = new Map<string, StreamableHttpServerTransport>();

// MCP endpoint
app.all("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"] as string;

  let transport: StreamableHttpServerTransport;

  if (sessionId && sessions.has(sessionId)) {
    transport = sessions.get(sessionId)!;
  } else {
    transport = new StreamableHttpServerTransport({
      sessionIdGenerator: () => crypto.randomUUID(),
    });
    await server.connect(transport);
  }

  await transport.handleRequest(req, res);

  // Store session if new
  const newSessionId = res.getHeader("Mcp-Session-Id") as string;
  if (newSessionId) {
    sessions.set(newSessionId, transport);
  }
});

app.listen(3000, () => {
  console.log("MCP HTTP Server running on port 3000");
});
```

---

## Testing and Debugging

### MCP Inspector

The **MCP Inspector** is the official testing tool for MCP servers. It's like Postman for MCP.

#### Installation & Usage

```bash
# Test a local stdio server
npx @modelcontextprotocol/inspector node build/server.js

# Test with arguments
npx @modelcontextprotocol/inspector npx tsx server.ts

# Test a Python server
npx @modelcontextprotocol/inspector python server.py
```

The Inspector opens at `http://localhost:6274` and provides:

- **Tools Panel**: List, call, and test all server tools
- **Resources Panel**: Browse and read resources
- **Prompts Panel**: Test prompt templates
- **Notifications Tab**: View server notifications
- **History Tab**: Review request/response history

#### Security Note

The Inspector generates a session token on startup. For remote access, use:

```bash
npx @modelcontextprotocol/inspector --token YOUR_TOKEN
```

### Manual Testing with stdio

You can test stdio servers manually by piping JSON-RPC messages:

```bash
# Initialize and list tools
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | node build/server.js
```

### Debugging Tips

1. **Always log to stderr**, never stdout:
   ```typescript
   console.error("Debug:", data); // Correct
   console.log("Debug:", data);   // WRONG - breaks protocol
   ```

2. **Enable verbose logging**:
   ```typescript
   const server = new McpServer({
     name: "my-server",
     version: "1.0.0",
   }, {
     capabilities: {
       logging: {},
     },
   });
   ```

3. **Use the MCP Inspector's Network tab** to view raw JSON-RPC messages

4. **Validate JSON schemas** before deploying - malformed schemas break tool discovery

---

## Client Configuration

### Claude Desktop

#### Configuration File Location

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

#### Basic Configuration

```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/server/build/index.js"],
      "env": {
        "API_KEY": "your-api-key"
      }
    },
    "python-server": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "DATABASE_URL": "postgresql://..."
      }
    },
    "npx-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
    }
  }
}
```

#### Using Environment Variables

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Claude Code (CLI)

#### Adding Servers

```bash
# Add a stdio server
claude mcp add my-server -- node /path/to/server.js

# Add with environment variables
claude mcp add github -e GITHUB_TOKEN=ghp_xxx -- npx -y @modelcontextprotocol/server-github

# Add HTTP server
claude mcp add --transport http notion https://mcp.notion.com/mcp

# Add with authentication header
claude mcp add --transport http api-server https://api.example.com/mcp \
  --env API_KEY="secret" \
  --header "Authorization: Bearer \${API_KEY}"
```

#### Scope Options

```bash
# Local scope (default) - only you, only this project
claude mcp add --scope local my-server -- node server.js

# Project scope - shared via .mcp.json
claude mcp add --scope project shared-server -- node server.js

# User scope - available in all your projects
claude mcp add --scope user my-tools -- node tools.js
```

#### Managing Servers

```bash
# List all servers
claude mcp list

# Get server details
claude mcp get my-server

# Remove a server
claude mcp remove my-server
```

#### JSON Configuration

```bash
claude mcp add-json my-server '{
  "command": "node",
  "args": ["server.js"],
  "env": {"DEBUG": "true"}
}'
```

#### Windows Note

On Windows (not WSL), wrap npx commands with `cmd /c`:

```bash
claude mcp add my-server -- cmd /c npx -y @some/package
```

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "mcp.servers": {
    "my-server": {
      "command": "node",
      "args": ["${workspaceFolder}/mcp-server/build/index.js"]
    }
  }
}
```

### Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/absolute/path/to/server.js"]
    }
  }
}
```

---

## Security Best Practices

### Authentication

#### OAuth 2.1 (Recommended for HTTP)

```typescript
// server.ts with OAuth
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHttpServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const transport = new StreamableHttpServerTransport({
  authProvider: {
    // Implement OAuth 2.1 flow
    validateToken: async (token: string) => {
      const result = await validateWithAuthServer(token);
      return result.valid;
    },
    getAuthorizationUrl: () => "https://auth.example.com/authorize",
    getTokenUrl: () => "https://auth.example.com/token",
  },
});
```

#### Environment-Based (stdio)

```typescript
// Validate credentials from environment
const apiKey = process.env.API_KEY;
if (!apiKey) {
  console.error("API_KEY environment variable required");
  process.exit(1);
}
```

### Authorization

#### Role-Based Access Control

```typescript
server.tool(
  "admin_action",
  "Perform administrative action",
  { action: z.string() },
  async ({ action }, context) => {
    // Check user role
    const userRole = context.meta?.role;
    if (userRole !== "admin") {
      return {
        content: [{ type: "text", text: "Unauthorized: Admin role required" }],
        isError: true,
      };
    }
    // Perform action...
  }
);
```

#### Tool-Level Permissions

```typescript
const toolPermissions = {
  read_data: ["user", "admin"],
  write_data: ["admin"],
  delete_data: ["admin"],
};

function checkPermission(tool: string, role: string): boolean {
  return toolPermissions[tool]?.includes(role) ?? false;
}
```

### Input Validation

Always validate inputs at system boundaries:

```typescript
import { z } from "zod";

// Strict schema validation
server.tool(
  "query_database",
  "Execute safe database query",
  {
    table: z.enum(["users", "products", "orders"]), // Whitelist tables
    id: z.number().int().positive(), // Validate ID format
    fields: z.array(z.string().regex(/^[a-zA-Z_]+$/)).max(10), // Safe field names
  },
  async ({ table, id, fields }) => {
    // Safe to use - all inputs validated
  }
);
```

### Secret Management

```typescript
// NEVER do this
const config = {
  apiKey: "sk-hardcoded-secret", // WRONG
};

// DO this instead
const config = {
  apiKey: process.env.API_KEY,
};

// Or use secret management
import { SecretManagerServiceClient } from "@google-cloud/secret-manager";
const client = new SecretManagerServiceClient();
const [secret] = await client.accessSecretVersion({ name: "projects/.../secrets/api-key/versions/latest" });
```

### Context Isolation

```typescript
// Ensure each session gets fresh context
server.tool("process_data", "...", { data: z.string() }, async ({ data }, context) => {
  // Create isolated scope per request
  const sessionScope = createIsolatedScope(context.sessionId);

  try {
    return await sessionScope.process(data);
  } finally {
    // Clean up to prevent data leakage
    sessionScope.destroy();
  }
});
```

---

## Error Handling

### JSON-RPC Error Codes

MCP uses standard JSON-RPC 2.0 error codes plus custom ranges:

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse Error | Invalid JSON |
| -32600 | Invalid Request | Not a valid request object |
| -32601 | Method Not Found | Method doesn't exist |
| -32602 | Invalid Params | Invalid method parameters |
| -32603 | Internal Error | Internal server error |
| -32000 to -32099 | Server Error | Reserved for server errors |

### Error Response Format

```typescript
// Protocol-level error
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params: 'query' is required",
    "data": {
      "field": "query",
      "reason": "missing_required"
    }
  }
}

// Tool execution error (different pattern!)
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Database connection failed: timeout after 30s"
      }
    ],
    "isError": true  // Flag indicates tool failure
  }
}
```

### Best Practices

```typescript
server.tool(
  "risky_operation",
  "Operation that might fail",
  { input: z.string() },
  async ({ input }) => {
    try {
      const result = await performOperation(input);
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
      };
    } catch (error) {
      // Log full error server-side
      console.error("Operation failed:", error);

      // Return helpful message to AI (not system internals)
      return {
        content: [
          {
            type: "text",
            text: `Operation failed: ${error.message}. ` +
                  `Try: 1) Check input format, 2) Retry with smaller input, ` +
                  `3) Use alternative_operation tool instead.`,
          },
        ],
        isError: true,
      };
    }
  }
);
```

### Recovery-Oriented Errors

Help AI assistants recover gracefully:

```typescript
// Instead of generic errors
return { content: [{ type: "text", text: "Error occurred" }], isError: true };

// Provide actionable guidance
return {
  content: [{
    type: "text",
    text: JSON.stringify({
      error: "Rate limit exceeded",
      retryAfter: 60,
      suggestion: "Wait 60 seconds before retrying, or use batch_query tool for multiple items",
      alternativeTools: ["batch_query", "cached_query"]
    })
  }],
  isError: true,
};
```

---

## Deployment

### Local Development

```bash
# TypeScript
npx tsx watch server.ts

# Python
python server.py
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM node:20-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY build/ ./build/

# For stdio servers
CMD ["node", "build/index.js"]

# For HTTP servers
EXPOSE 3000
CMD ["node", "build/http-server.js"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-server:
    build: .
    environment:
      - API_KEY=${API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    ports:
      - "3000:3000"
```

### Cloud Platforms

#### AWS Lambda (Streamable HTTP)

```typescript
// lambda.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHttpServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const server = new McpServer({ name: "lambda-mcp", version: "1.0.0" });
// ... define tools ...

export const handler = async (event: any, context: any) => {
  const transport = new StreamableHttpServerTransport();
  await server.connect(transport);
  return transport.handleLambdaEvent(event);
};
```

#### Google Cloud Run

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm ci && npm run build
EXPOSE 8080
CMD ["node", "build/http-server.js"]
```

```bash
gcloud run deploy mcp-server \
  --source . \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="API_KEY=xxx"
```

### Package Distribution

#### npm Package

```json
{
  "name": "@yourorg/mcp-server-example",
  "version": "1.0.0",
  "bin": {
    "mcp-server-example": "./build/index.js"
  },
  "files": ["build"],
  "scripts": {
    "build": "tsc",
    "prepublishOnly": "npm run build"
  }
}
```

#### PyPI Package

```toml
# pyproject.toml
[project]
name = "mcp-server-example"
version = "1.0.0"

[project.scripts]
mcp-server-example = "mcp_server_example:main"
```

---

## Resources

### Official Documentation

- [MCP Specification](https://modelcontextprotocol.io/specification/2025-06-18/server)
- [MCP Developer Docs](https://modelcontextprotocol.io/docs/develop/build-server)
- [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)

### Example Servers

- [Official Server Repository](https://github.com/modelcontextprotocol/servers)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)
- [Community Servers](https://github.com/punkpeye/awesome-mcp-servers)

### Tutorials

- [Anthropic MCP Course](https://anthropic.skilljar.com/introduction-to-model-context-protocol)
- [Hugging Face MCP Course](https://huggingface.co/learn/mcp-course)
- [DataCamp MCP Tutorial](https://www.datacamp.com/tutorial/mcp-model-context-protocol)
- [FreeCodeCamp TypeScript Guide](https://www.freecodecamp.org/news/how-to-build-a-custom-mcp-server-with-typescript-a-handbook-for-developers/)

### Client Integration Guides

- [Claude Desktop Setup](https://modelcontextprotocol.io/docs/develop/connect-local-servers)
- [Claude Code MCP Docs](https://code.claude.com/docs/en/mcp)
- [VS Code MCP Setup](https://code.visualstudio.com/docs/copilot/customization/mcp-servers)

### Security Resources

- [MCP Authorization Guide](https://modelcontextprotocol.io/docs/tutorials/security/authorization)
- [MCP Security Best Practices](https://tetrate.io/learn/ai/mcp/security-privacy-considerations)

---

## Quick Reference

### Initialization Sequence

```
Client → Server: initialize (protocol version, capabilities, client info)
Server → Client: initialize response (protocol version, capabilities, server info)
Client → Server: initialized notification
--- Normal operations begin ---
```

### Capability Negotiation

```json
{
  "capabilities": {
    "tools": {},
    "resources": { "subscribe": true },
    "prompts": {},
    "logging": {}
  }
}
```

### Common Methods

| Method | Direction | Description |
|--------|-----------|-------------|
| `initialize` | Client → Server | Start session |
| `initialized` | Client → Server | Confirm ready |
| `tools/list` | Client → Server | List available tools |
| `tools/call` | Client → Server | Execute a tool |
| `resources/list` | Client → Server | List resources |
| `resources/read` | Client → Server | Read resource content |
| `prompts/list` | Client → Server | List prompts |
| `prompts/get` | Client → Server | Get prompt messages |
| `ping` | Bidirectional | Health check |

### Tool Result Format

```typescript
{
  content: [
    { type: "text", text: "Result text" },
    { type: "image", data: "base64...", mimeType: "image/png" },
    { type: "resource", resource: { uri: "file://...", text: "..." } }
  ],
  isError?: boolean  // true if tool execution failed
}
```

---

*Last updated: January 2025*
*MCP Specification Version: 2025-06-18*
