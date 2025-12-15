# Kotlin MCP Server with Ktor

A Model Context Protocol (MCP) server implementation in Kotlin using Ktor framework.

## Features

- **JSON-RPC 2.0** compliant
- **MCP Protocol** support (version 2024-11-05)
- Three test tools included:
  - `test_tool` - Echoes back a message with optional uppercase conversion
  - `get_time` - Returns the current server time
  - `calculator` - Performs basic arithmetic operations

## Requirements

- JDK 17 or higher
- Gradle 8.5 (included via wrapper)

## Running the Server

```bash
./gradlew run
```

The server will start on `http://localhost:8080`

## Endpoints

- `GET /` - Server info
- `GET /health` - Health check
- `POST /mcp` - MCP JSON-RPC endpoint
- `POST /` - Alternative MCP endpoint

## Example Requests

### Initialize

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}'
```

### List Tools

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

### Call Test Tool

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"test_tool","arguments":{"message":"Hello World!","uppercase":true}}}'
```

### Call Calculator

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"calculator","arguments":{"operation":"multiply","a":6,"b":7}}}'
```

### Get Time

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_time","arguments":{}}}'
```

## Building

```bash
./gradlew build
```

## Creating a Fat JAR

```bash
./gradlew buildFatJar
```

The JAR will be created in `build/libs/mcp-server-kotlin-all.jar`

