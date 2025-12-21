# Telegram bot + Mobile MCP (mobile-next/mobile-mcp) integration

This folder contains a Telegram bot (`simple_bot.py`) that can:

- Talk to existing HTTP MCP servers (Calendar + Events) already in this repo
- Start and control **Mobile MCP** via stdio MCP: `npx -y @mobilenext/mobile-mcp@latest`
- Start an Android emulator (AVD) or boot an iOS Simulator, then call Mobile MCP tools to interact (tap, screenshot, etc.)

Mobile MCP project: [mobile-next/mobile-mcp](https://github.com/mobile-next/mobile-mcp)

## Prerequisites

Per Mobile MCP README you typically need:

- Node.js v22+ (for `npx`)
- Android Platform Tools / Emulator (for Android)
- Xcode command line tools + Simulator (for iOS)

See: [mobile-next/mobile-mcp](https://github.com/mobile-next/mobile-mcp)

## Environment variables

Required (existing bot):

- `YANDEX_FOLDER_ID`
- `YANDEX_AUTH`
- `TELEGRAM_BOT_TOKEN`

Optional (Mobile MCP integration):

- `MOBILE_MCP_COMMAND` (default: `npx -y @mobilenext/mobile-mcp@latest`)
- `ANDROID_EMULATOR_BIN` (default: `emulator`) — you can set this in `.env` (recommended on macOS), e.g. `~/Library/Android/sdk/emulator/emulator`

## Run

From the repo root:

```bash
source new-env/bin/activate
python3 telegram_chat_bot/simple_bot.py
```

## Telegram commands (Mobile MCP)

- `/mobile_start` — start Mobile MCP (`MOBILE_MCP_COMMAND`)
- `/mobile_status` — show whether the Mobile MCP process is running + stderr tail
- `/mobile_tools` — list tools provided by Mobile MCP
- `/mobile_devices` — list available devices (device ids)
- `/mobile_use <device>` — select device for this chat (so `/mobile_call` can auto-inject `device`)
- `/mobile_call <tool> [json|k=v]` — call any tool
  - Example: `/mobile_call tap {"x":120,"y":640}`
  - Example: `/mobile_call tap x=120 y=640`

Convenience:

- `/tap <x> <y>` — tries to find a tap tool (`tap/click/touch/...`) and calls it
- `/screenshot` — tries to find a screenshot tool and sends the image(s) to Telegram

## Emulator commands

Android:

- `/android_avds` — list available AVDs (uses `emulator -list-avds`)
- `/android_boot <avd> [headless]` — start emulator process
- `/android_stop` — stop emulator process started by the bot

iOS Simulator:

- `/ios_devices` — list available iOS Simulator devices
- `/ios_boot <name|udid>` — boot a device
- `/ios_open` — open the Simulator app UI

## Notes / troubleshooting

- Mobile MCP is started on-demand (first `/mobile_start` or any `/mobile_*` call).
- If `/mobile_start` fails, check `/mobile_status` for stderr tail.
- Tool names differ by Mobile MCP version and platform; use `/mobile_tools` to discover the exact tool names.
- If `/android_boot` fails with errors like **Qt library not found** / **qemu-system-... not found**, it usually means the bot is launching a wrong `emulator` binary.
  - Ensure you have Android SDK Emulator installed and set `ANDROID_EMULATOR_BIN` to the full path, e.g. `~/Library/Android/sdk/emulator/emulator`
  - Use `/diag` to see what paths the bot detects.


