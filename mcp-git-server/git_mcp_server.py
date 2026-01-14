#!/usr/bin/env python3
"""
MCP Server for Git Repository Integration.
Provides tools for reading repository info, files, and commits.
Communicates via JSON-RPC 2.0 over stdio.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Default repository path (can be overridden via env)
REPO_PATH = os.environ.get("GIT_REPO_PATH", os.path.join(os.path.dirname(__file__), "..", "bookechi_repo"))


def git_cmd(args: List[str], repo_path: str = None) -> tuple[bool, str]:
    """Execute git command and return (success, output)."""
    repo_path = repo_path or REPO_PATH
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


# === Tool Implementations ===

def tool_get_branch() -> dict:
    """Get current git branch."""
    success, output = git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
    if success:
        return {"branch": output}
    return {"error": output}


def tool_get_status() -> dict:
    """Get git status (changed files)."""
    result = {"staged": [], "modified": [], "untracked": []}
    
    # Staged files
    success, output = git_cmd(["diff", "--cached", "--name-only"])
    if success and output:
        result["staged"] = output.split("\n")
    
    # Modified files
    success, output = git_cmd(["diff", "--name-only"])
    if success and output:
        result["modified"] = output.split("\n")
    
    # Untracked files
    success, output = git_cmd(["ls-files", "--others", "--exclude-standard"])
    if success and output:
        result["untracked"] = output.split("\n")
    
    return result


def tool_get_log(count: int = 10) -> dict:
    """Get recent commits."""
    count = max(1, min(count, 50))
    success, output = git_cmd(["log", f"-{count}", "--pretty=format:%H|%an|%ae|%ar|%s"])
    
    if not success:
        return {"error": output}
    
    commits = []
    for line in output.split("\n"):
        if not line:
            continue
        parts = line.split("|", 4)
        if len(parts) >= 5:
            commits.append({
                "hash": parts[0][:8],
                "full_hash": parts[0],
                "author": parts[1],
                "email": parts[2],
                "date": parts[3],
                "message": parts[4]
            })
    
    return {"commits": commits, "count": len(commits)}


def tool_get_diff(file_path: str = None, commit: str = None) -> dict:
    """Get diff for a file or commit."""
    args = ["diff"]
    if commit:
        args.append(commit)
    if file_path:
        args.extend(["--", file_path])
    
    success, output = git_cmd(args)
    if success:
        # Truncate large diffs
        if len(output) > 10000:
            output = output[:10000] + "\n... (truncated)"
        return {"diff": output}
    return {"error": output}


def tool_read_file(path: str) -> dict:
    """Read file content from repository."""
    full_path = os.path.join(REPO_PATH, path)
    
    if not os.path.exists(full_path):
        return {"error": f"File not found: {path}"}
    
    if not os.path.isfile(full_path):
        return {"error": f"Not a file: {path}"}
    
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        # Truncate very large files
        if len(content) > 50000:
            content = content[:50000] + "\n... (truncated, file too large)"
        
        return {
            "path": path,
            "content": content,
            "size": os.path.getsize(full_path)
        }
    except Exception as e:
        return {"error": str(e)}


def tool_list_files(directory: str = "", extension: str = None, max_depth: int = 3) -> dict:
    """List files in repository directory."""
    target_dir = os.path.join(REPO_PATH, directory) if directory else REPO_PATH
    
    if not os.path.isdir(target_dir):
        return {"error": f"Directory not found: {directory}"}
    
    files = []
    dirs = []
    
    try:
        for root, dirnames, filenames in os.walk(target_dir):
            # Calculate depth
            rel_root = os.path.relpath(root, target_dir)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            
            if depth > max_depth:
                continue
            
            # Skip hidden and common ignored directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ["build", "__pycache__", "node_modules"]]
            
            for filename in filenames:
                if filename.startswith("."):
                    continue
                if extension and not filename.endswith(extension):
                    continue
                
                rel_path = os.path.relpath(os.path.join(root, filename), REPO_PATH)
                files.append(rel_path)
            
            for dirname in dirnames:
                rel_path = os.path.relpath(os.path.join(root, dirname), REPO_PATH)
                dirs.append(rel_path)
        
        return {
            "directory": directory or "/",
            "files": files[:200],
            "directories": dirs[:100],
            "total_files": len(files),
            "total_dirs": len(dirs)
        }
    except Exception as e:
        return {"error": str(e)}


def tool_search_code(pattern: str, file_extension: str = None, max_results: int = 20) -> dict:
    """Search for pattern in code files."""
    args = ["grep", "-r", "-n", "-I", "--include=*.kt", "--include=*.java", "--include=*.xml", "--include=*.md", pattern]
    
    if file_extension:
        args = ["grep", "-r", "-n", "-I", f"--include=*{file_extension}", pattern]
    
    success, output = git_cmd(args)
    
    if not success and not output:
        return {"matches": [], "count": 0}
    
    matches = []
    for line in output.split("\n")[:max_results]:
        if ":" in line:
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "file": parts[0],
                    "line": int(parts[1]) if parts[1].isdigit() else 0,
                    "content": parts[2][:200]
                })
    
    return {"pattern": pattern, "matches": matches, "count": len(matches)}


def tool_get_file_history(path: str, count: int = 5) -> dict:
    """Get commit history for a specific file."""
    count = max(1, min(count, 20))
    success, output = git_cmd(["log", f"-{count}", "--pretty=format:%H|%an|%ar|%s", "--", path])
    
    if not success:
        return {"error": output}
    
    commits = []
    for line in output.split("\n"):
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) >= 4:
            commits.append({
                "hash": parts[0][:8],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3]
            })
    
    return {"file": path, "commits": commits}


# === MCP Protocol ===

TOOLS = [
    {
        "name": "git_branch",
        "description": "Get current git branch name",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "git_status",
        "description": "Get git status - staged, modified, and untracked files",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "git_log",
        "description": "Get recent git commits",
        "inputSchema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of commits to return (default: 10, max: 50)"
                }
            },
            "required": []
        }
    },
    {
        "name": "git_diff",
        "description": "Get diff for a file or commit",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file (optional)"
                },
                "commit": {
                    "type": "string",
                    "description": "Commit hash to compare (optional)"
                }
            },
            "required": []
        }
    },
    {
        "name": "read_file",
        "description": "Read file content from the repository",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file relative to repository root"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List files and directories in the repository",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path (default: root)"
                },
                "extension": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.kt')"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth (default: 3)"
                }
            },
            "required": []
        }
    },
    {
        "name": "search_code",
        "description": "Search for pattern in source code files",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (grep syntax)"
                },
                "file_extension": {
                    "type": "string",
                    "description": "Filter by extension (e.g., '.kt')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results (default: 20)"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "file_history",
        "description": "Get commit history for a specific file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of commits (default: 5)"
                }
            },
            "required": ["path"]
        }
    }
]


def execute_tool(name: str, args: Dict[str, Any]) -> dict:
    """Execute a tool by name with given arguments."""
    tool_map = {
        "git_branch": lambda: tool_get_branch(),
        "git_status": lambda: tool_get_status(),
        "git_log": lambda: tool_get_log(args.get("count", 10)),
        "git_diff": lambda: tool_get_diff(args.get("file_path"), args.get("commit")),
        "read_file": lambda: tool_read_file(args.get("path", "")),
        "list_files": lambda: tool_list_files(
            args.get("directory", ""),
            args.get("extension"),
            args.get("max_depth", 3)
        ),
        "search_code": lambda: tool_search_code(
            args.get("pattern", ""),
            args.get("file_extension"),
            args.get("max_results", 20)
        ),
        "file_history": lambda: tool_get_file_history(
            args.get("path", ""),
            args.get("count", 5)
        )
    }
    
    if name not in tool_map:
        return {"error": f"Unknown tool: {name}"}
    
    return tool_map[name]()


def handle_request(request: dict) -> Optional[dict]:
    """Handle a JSON-RPC request."""
    method = request.get("method", "")
    request_id = request.get("id")
    params = request.get("params", {})
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "git-mcp-server",
                    "version": "1.0.0"
                }
            }
        }
    
    if method in ("notifications/initialized", "initialized"):
        # Notification - no response needed if no id
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": TOOLS}
        }
    
    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        
        result = execute_tool(tool_name, tool_args)
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, indent=2)
                    }
                ]
            }
        }
    
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}"
        }
    }


def main():
    """Main entry point - read from stdin, write to stdout."""
    sys.stderr.write(f"Git MCP Server started. Repo: {REPO_PATH}\n")
    sys.stderr.flush()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            request = json.loads(line)
            response = handle_request(request)
            
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {e}"
                }
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()


