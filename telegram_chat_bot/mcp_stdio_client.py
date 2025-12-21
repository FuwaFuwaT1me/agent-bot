#!/usr/bin/env python3
"""
Minimal MCP stdio (JSON-RPC over stdin/stdout) client.

Used to integrate Node-based MCP servers such as Mobile MCP:
`npx -y @mobilenext/mobile-mcp@latest`
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class McpStdioError(RuntimeError):
    pass


@dataclass
class McpServerInfo:
    protocol_version: str
    name: str
    version: str
    capabilities: Dict[str, Any]


class McpStdioClient:
    """
    JSON-RPC 2.0 client talking to MCP server via stdio, one JSON per line.
    """

    def __init__(
        self,
        command: List[str],
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self.command = command
        self.env = env
        self.cwd = cwd

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

        self._request_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()

        self._stderr_ring: List[str] = []
        self._stderr_ring_max = 200

        self._initialized = False

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def start(self) -> None:
        if self.is_running:
            return

        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
        )

        if not self._proc.stdin or not self._proc.stdout or not self._proc.stderr:
            raise McpStdioError("Failed to create stdio pipes for MCP server process")

        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr_loop())
        self._initialized = False

    async def stop(self) -> None:
        self._initialized = False

        # Cancel reader tasks first to stop awaiting on pipes.
        for t in (self._reader_task, self._stderr_task):
            if t and not t.done():
                t.cancel()

        # Fail any pending requests.
        for _id, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(McpStdioError("MCP server stopped"))
        self._pending.clear()

        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()

        self._proc = None
        self._reader_task = None
        self._stderr_task = None

    def get_recent_stderr(self) -> str:
        return "".join(self._stderr_ring[-self._stderr_ring_max :])

    async def _read_stdout_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8", errors="replace").strip())
            except Exception:
                continue

            # Responses have an "id". Notifications may not.
            msg_id = msg.get("id")
            if msg_id is None:
                continue

            fut = self._pending.pop(int(msg_id), None)
            if fut is None or fut.done():
                continue

            if "error" in msg and msg["error"]:
                err = msg["error"]
                fut.set_exception(
                    McpStdioError(f"{err.get('message', 'MCP error')} ({err})")
                )
            else:
                fut.set_result(msg.get("result"))

    async def _read_stderr_loop(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            s = line.decode("utf-8", errors="replace")
            self._stderr_ring.append(s)
            if len(self._stderr_ring) > self._stderr_ring_max:
                self._stderr_ring = self._stderr_ring[-self._stderr_ring_max :]

    async def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None, *, timeout_s: float = 60
    ) -> Any:
        if not self.is_running:
            raise McpStdioError("MCP server is not running")
        assert self._proc and self._proc.stdin

        req_id = self._next_id()
        request: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            request["params"] = params

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        payload = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()

        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            self._pending.pop(req_id, None)
            raise McpStdioError(f"Timeout calling MCP method {method}") from e

    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_running:
            raise McpStdioError("MCP server is not running")
        assert self._proc and self._proc.stdin

        msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params

        payload = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()

    async def initialize(self) -> McpServerInfo:
        """
        Initialize MCP session.
        MCP requires sending notifications/initialized after initialize succeeds.
        """
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "telegram_chat_bot", "version": "1.0"},
            },
            timeout_s=60,
        )
        await self._send_notification("notifications/initialized", {})
        self._initialized = True

        protocol_version = (result or {}).get("protocolVersion", "unknown")
        server_info = (result or {}).get("serverInfo", {}) or {}
        capabilities = (result or {}).get("capabilities", {}) or {}

        return McpServerInfo(
            protocol_version=str(protocol_version),
            name=str(server_info.get("name", "unknown")),
            version=str(server_info.get("version", "unknown")),
            capabilities=capabilities,
        )

    async def list_tools(self) -> List[Dict[str, Any]]:
        result = await self._send_request("tools/list", timeout_s=30)
        return (result or {}).get("tools", []) or []

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        result = await self._send_request("tools/call", params, timeout_s=120)
        return result or {}


