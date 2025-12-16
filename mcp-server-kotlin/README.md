# Pokemon MCP Server (Kotlin)

A Model Context Protocol (MCP) server with Pokemon data from [PokéAPI](https://pokeapi.co).

## Two Transport Modes

### 1. STDIO Mode (for Claude Desktop, Cursor)
```bash
java -jar build/libs/mcp-server-kotlin-1.0.0.jar
```
This is the "real" MCP way - communicates via stdin/stdout.

### 2. HTTP Mode (for Telegram bot, curl testing)
```bash
java -jar build/libs/mcp-server-kotlin-1.0.0.jar --http 8080
```
Starts an HTTP server on port 8080.

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_pokemon` | Get Pokemon stats, types, abilities | `name` |
| `get_type` | Get type damage relations | `name` |
| `get_move` | Get move power, accuracy, effect | `name` |
| `get_ability` | Get ability effect | `name` |
| `list_pokemon` | List Pokemon with pagination | `limit`, `offset` |

## Building

```bash
cd mcp-server-kotlin
./gradlew jar
```

## Usage with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pokemon": {
      "command": "java",
      "args": ["-jar", "/path/to/mcp-server-kotlin-1.0.0.jar"]
    }
  }
}
```

## Usage with Telegram Bot

1. Start the server in HTTP mode:
```bash
java -jar build/libs/mcp-server-kotlin-1.0.0.jar --http 8080
```

2. Your bot connects to `http://localhost:8080/mcp`

3. Use bot commands:
```
/mcp_tools
/mcp_call get_pokemon {"name": "pikachu"}
/mcp_call get_type {"name": "fire"}
```

## Testing with curl

```bash
# Health check
curl http://localhost:8080/health

# List tools
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# Get Pokemon
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_pokemon","arguments":{"name":"pikachu"}}}'
```

## Project Structure

```
src/main/kotlin/com/example/mcp/
├── Main.kt           # Entry point (stdio + http modes)
├── McpServer.kt      # JSON-RPC message handling  
├── McpProtocol.kt    # MCP protocol data types
├── PokeApiTools.kt   # Tool definitions & execution
└── PokeApiClient.kt  # HTTP client for PokéAPI
```
