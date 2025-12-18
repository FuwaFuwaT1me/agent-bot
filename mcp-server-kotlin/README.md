# Yandex Calendar MCP Server (Kotlin)

A Model Context Protocol (MCP) server for Yandex Calendar integration.

## Features

- Get today's events
- Get upcoming events (next N days)
- Get events for specific date
- Create new events
- Daily summary with greeting

## Setup

### 1. Create Yandex App Password

1. Go to https://id.yandex.ru/security/app-passwords
2. Click "Create app password"
3. Select "Calendar" as the app type
4. Copy the generated password

### 2. Set Environment Variables

```bash
export YANDEX_USERNAME="your-yandex-login"
export YANDEX_APP_PASSWORD="your-app-password"
```

### 3. Build

```bash
cd mcp-server-kotlin
./gradlew jar
```

## Usage

### HTTP Mode (for Telegram bot)

```bash
java -jar build/libs/mcp-server-kotlin-1.0.0.jar --http 8080
```

### STDIO Mode (for Claude Desktop)

```bash
java -jar build/libs/mcp-server-kotlin-1.0.0.jar
```

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_today_events` | Get all events for today | none |
| `get_upcoming_events` | Get events for next N days | `days` (optional, default: 7) |
| `get_events_for_date` | Get events for specific date | `date` (YYYY-MM-DD) |
| `create_event` | Create a new event | `title`, `date`, `start_time`, `end_time`, `description` |
| `get_daily_summary` | Get formatted daily summary | none |

## Telegram Bot Commands

```
/mcp_tools - List available tools
/mcp_call get_today_events
/mcp_call get_upcoming_events 7
/mcp_call get_events_for_date 2024-12-25
/mcp_call get_daily_summary
/mcp_call create_event Meeting 2024-12-20 14:00 15:00
```

## Claude Desktop Config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "calendar": {
      "command": "java",
      "args": ["-jar", "/path/to/mcp-server-kotlin-1.0.0.jar"],
      "env": {
        "YANDEX_USERNAME": "your-login",
        "YANDEX_APP_PASSWORD": "your-app-password"
      }
    }
  }
}
```

## Testing with curl

```bash
# Get today's events
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_today_events","arguments":{}}}'

# Get daily summary
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_daily_summary","arguments":{}}}'
```
