#!/usr/bin/env python3
"""
Mobile MCP integration helpers:
- start/stop Mobile MCP server (Node, stdio MCP)
- optional emulator control (Android/iOS) from the same Telegram bot process
"""

from __future__ import annotations

import asyncio
import base64
import os
import shlex
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mcp_stdio_client import McpStdioClient, McpStdioError, McpServerInfo


def _env_dict_with_os(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return merged


def pick_tool_name(tools: List[Dict[str, Any]], candidates: List[str]) -> Optional[str]:
    available = {t.get("name", "") for t in tools}
    for c in candidates:
        if c in available:
            return c
    # Try case-insensitive match
    lower_map = {str(name).lower(): name for name in available}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def parse_kv_args(arg_str: str) -> Dict[str, Any]:
    """
    Parses "k=v k2=v2" into a dict. Values are kept as strings (best-effort coercion).
    """
    out: Dict[str, Any] = {}
    if not arg_str.strip():
        return out
    for token in shlex.split(arg_str):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        # Best-effort coercion
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                if "." in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def extract_images_from_mcp_result(result: Dict[str, Any]) -> List[Tuple[bytes, str]]:
    """
    Extracts image bytes from MCP tool result content.
    MCP often uses content items like:
      {"type":"image","data":"<base64>","mimeType":"image/png"}
    Returns list of (bytes, mimeType).
    """
    images: List[Tuple[bytes, str]] = []
    content = result.get("content", []) or []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image":
            continue
        data_b64 = item.get("data")
        mime = item.get("mimeType", "application/octet-stream")
        if not data_b64:
            continue
        try:
            raw = base64.b64decode(data_b64)
        except Exception:
            continue
        images.append((raw, str(mime)))
    return images


def extract_text_from_mcp_result(result: Dict[str, Any]) -> str:
    content = result.get("content", []) or []
    chunks: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            chunks.append(str(item.get("text", "")))
    return "\n".join([c for c in chunks if c]).strip()


@dataclass
class EmulatorState:
    android_proc: Optional[asyncio.subprocess.Process] = None
    android_avd: Optional[str] = None
    android_last_error: str = ""


class MobileMcpService:
    """
    Singleton-like service object for Mobile MCP server process (stdio MCP).
    """

    def __init__(self, command: Optional[List[str]] = None) -> None:
        # Default matches Mobile MCP README:
        # npx -y @mobilenext/mobile-mcp@latest
        self.command = command or ["npx", "-y", "@mobilenext/mobile-mcp@latest"]
        self.client = McpStdioClient(self.command, env=_env_dict_with_os())
        self._start_lock = asyncio.Lock()

        self.emulator = EmulatorState()

    async def ensure_started(self) -> McpServerInfo:
        async with self._start_lock:
            if not self.client.is_running:
                await self.client.start()
            if not self.client.initialized:
                return await self.client.initialize()
            # If already initialized, best-effort return status
            return McpServerInfo(
                protocol_version="unknown",
                name="mobile-mcp",
                version="unknown",
                capabilities={},
            )

    async def stop(self) -> None:
        await self.client.stop()

    async def list_tools(self) -> List[Dict[str, Any]]:
        await self.ensure_started()
        return await self.client.list_tools()

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        await self.ensure_started()
        return await self.client.call_tool(name, arguments)

    def recent_stderr(self) -> str:
        return self.client.get_recent_stderr()

    # === Emulator control ===
    async def android_list_avds(self) -> List[str]:
        """
        Uses `emulator -list-avds` if available.
        """
        emulator_bin = os.getenv("ANDROID_EMULATOR_BIN", "emulator")
        emulator_path = shutil.which(emulator_bin) or emulator_bin
        proc = await asyncio.create_subprocess_exec(
            emulator_path,
            "-list-avds",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(emulator_path) if os.path.isabs(emulator_path) else None,
        )
        out, _err = await proc.communicate()
        if proc.returncode != 0:
            return []
        return [line.strip() for line in out.decode("utf-8", errors="replace").splitlines() if line.strip()]

    async def android_boot(self, avd_name: str, *, headless: bool = False) -> str:
        """
        Boots Android emulator in background.
        """
        if self.emulator.android_proc and self.emulator.android_proc.returncode is None:
            return f"Android emulator already running (AVD={self.emulator.android_avd})"

        emulator_bin = os.getenv("ANDROID_EMULATOR_BIN", "emulator")
        # Prefer absolute path so emulator can locate its sibling folders (qt, qemu, etc).
        emulator_path = shutil.which(emulator_bin) or emulator_bin
        args = [emulator_path, "-avd", avd_name, "-no-snapshot-save"]
        if headless:
            args += ["-no-window"]

        self.emulator.android_last_error = ""
        try:
            # Setting cwd to emulator directory fixes cases where emulator tries to resolve
            # qt/qemu folders relative to current working directory.
            cwd = os.path.dirname(emulator_path) if os.path.isabs(emulator_path) else None
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError:
            return (
                "❌ Android emulator binary not found.\n"
                "Set ANDROID_EMULATOR_BIN or add Android emulator to PATH."
            )

        # Give it a moment to fail-fast if args/AVD are invalid.
        await asyncio.sleep(1.0)
        if proc.returncode is not None and proc.returncode != 0:
            out, err = await proc.communicate()
            self.emulator.android_last_error = (
                (err.decode("utf-8", errors="replace") if err else "")
                or (out.decode("utf-8", errors="replace") if out else "")
            ).strip()
            self.emulator.android_proc = None
            self.emulator.android_avd = None
            tail = (self.emulator.android_last_error or "unknown error")[-1500:]
            return f"❌ Android emulator failed to start (exit={proc.returncode}).\n\n{tail}"

        self.emulator.android_proc = proc
        self.emulator.android_avd = avd_name
        return (
            f"Android emulator boot started (AVD={avd_name}, headless={headless}).\n"
            f"{'⚠️ Headless mode: no window will appear.' if headless else ''}"
        ).strip()

    async def android_stop(self) -> str:
        proc = self.emulator.android_proc
        if not proc or proc.returncode is not None:
            self.emulator.android_proc = None
            self.emulator.android_avd = None
            return "Android emulator is not running"

        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=8)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        self.emulator.android_proc = None
        self.emulator.android_avd = None
        return "Android emulator stopped"

    async def ios_list_devices(self) -> str:
        """
        Returns `xcrun simctl list devices available` output (shortened).
        """
        proc = await asyncio.create_subprocess_exec(
            "xcrun",
            "simctl",
            "list",
            "devices",
            "available",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            return (err.decode("utf-8", errors="replace") or "").strip() or "Failed to list iOS devices"
        text = out.decode("utf-8", errors="replace").strip()
        # Keep it from becoming huge
        return text[:3500]

    async def ios_boot(self, device_name_or_udid: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "xcrun",
            "simctl",
            "boot",
            device_name_or_udid,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            return (err.decode("utf-8", errors="replace") or out.decode("utf-8", errors="replace")).strip() or "Failed to boot iOS Simulator"
        return f"iOS Simulator booted: {device_name_or_udid}"

    async def ios_open_simulator_app(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "open",
            "-a",
            "Simulator",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            return (err.decode("utf-8", errors="replace") or out.decode("utf-8", errors="replace")).strip() or "Failed to open Simulator app"
        return "Opened Simulator app"


async def safe_call(service: MobileMcpService, tool: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        return await service.call_tool(tool, args)
    except McpStdioError as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}


