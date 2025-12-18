# KudaGo Events MCP Server (Kotlin)

A Model Context Protocol (MCP) server for KudaGo API integration. Search for concerts, theater, exhibitions, festivals and more in Russian cities.

## Features

- 🔍 Search events by city, category, or query
- 🎫 Get detailed event information
- 🏟️ Search venues
- 🎵 Get upcoming concerts
- 📅 Get events formatted for calendar import (chain with Calendar MCP!)
- 🌍 List available cities and categories

## Why KudaGo?

- ✅ **No API key required** - completely free!
- ✅ **No geo-restrictions** - works from Russia!
- ✅ **Real events data** - actual concerts, theater, exhibitions
- ✅ **Russian cities** - Moscow, Saint Petersburg, Kazan, etc.

## Setup

### No setup required! 

Just build and run - no API keys needed.

### Build

```bash
cd mcp-ticketmaster-kotlin
./gradlew jar
```

## Usage

### HTTP Mode (for Telegram bot)

```bash
java -jar build/libs/mcp-ticketmaster-kotlin-1.0.0.jar --http 8081
```

### STDIO Mode (for Claude Desktop)

```bash
java -jar build/libs/mcp-ticketmaster-kotlin-1.0.0.jar
```

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_events` | Search events by city, category, query | `city`, `category`, `query`, `days_ahead`, `limit` |
| `get_event_details` | Get detailed event info | `event_id` (required) |
| `search_venues` | Search for venues | `city`, `query`, `limit` |
| `get_upcoming_concerts` | Get concerts in a city | `city`, `days_ahead`, `limit` |
| `get_events_for_calendar` | Events formatted for calendar import | `city`, `category`, `limit` |
| `list_cities` | List available cities | none |
| `list_categories` | List event categories | none |

### Available Cities

- Moscow (`msk`)
- Saint Petersburg (`spb`)
- Yekaterinburg (`ekb`)
- Kazan (`kzn`)
- Nizhny Novgorod (`nnv`)

### Event Categories

- `concert` - Concerts
- `theater` - Theater, opera, ballet
- `exhibition` - Exhibitions
- `festival` - Festivals
- `party` - Parties
- `kids` - Kids events
- And more...

## Telegram Bot Commands

```
/pipeline concert Moscow
/pipeline theater spb
/pipeline exhibition Kazan
/pipeline_status - check MCP servers
/pipeline_add 1 - add event #1 to calendar
/pipeline_add_all - add all events to calendar
```

## MCP Chaining Example

**Use with Calendar MCP to add events to your calendar:**

1. Search for events:
```
/pipeline concert Moscow
```

2. Add to calendar:
```
/pipeline_add 1
```

Or add all:
```
/pipeline_add_all
```

## Testing with curl

```bash
# Search concerts in Moscow
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_events","arguments":{"city":"Moscow","category":"concert","limit":5}}}'

# Get upcoming concerts in Saint Petersburg
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_upcoming_concerts","arguments":{"city":"spb","days_ahead":30}}}'

# List available cities
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_cities","arguments":{}}}'

# List all tools
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## Running Both MCP Servers

To chain Calendar and KudaGo MCPs:

```bash
# Terminal 1: Calendar MCP (port 8080)
cd mcp-server-kotlin
java -jar build/libs/mcp-server-kotlin-1.0.0.jar --http 8080

# Terminal 2: KudaGo Events MCP (port 8081)
cd mcp-ticketmaster-kotlin
java -jar build/libs/mcp-ticketmaster-kotlin-1.0.0.jar --http 8081

# Terminal 3: Telegram Bot
cd telegram_chat_bot
python simple_bot.py
```

## API Reference

KudaGo API Documentation: https://docs.kudago.com/
