"""
Local File System Agent — Powered by Claude API
================================================
Capabilities:
  - List & browse directories
  - Read & summarize files
  - Search by filename or content
  - Write / edit files
  - Automate multi-step workflows

Usage:
  python agent.py

Requirements:
  pip install anthropic
  Set ANTHROPIC_API_KEY environment variable
"""

import os
import re
import json
import fnmatch
import anthropic
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# 🔒 SAFETY: Restrict the agent to a specific root folder
# Change this to the folder you want the agent to access
ALLOWED_ROOT = Path(os.path.expanduser("~/agent_workspace")).resolve()
ALLOWED_ROOT.mkdir(parents=True, exist_ok=True)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_AGENTIC_LOOPS = 20   # prevent infinite loops

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────
# SAFETY HELPER
# ─────────────────────────────────────────────

def safe_path(raw_path: str) -> Path:
    """Resolve path and ensure it stays within ALLOWED_ROOT."""
    resolved = (ALLOWED_ROOT / raw_path).resolve()
    if not str(resolved).startswith(str(ALLOWED_ROOT)):
        raise PermissionError(
            f"Access denied: '{raw_path}' is outside the allowed folder '{ALLOWED_ROOT}'"
        )
    return resolved


# ─────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────

def list_directory(path: str = ".") -> dict:
    """List files and subdirectories at the given path."""
    try:
        target = safe_path(path)
        if not target.exists():
            return {"error": f"Path does not exist: {path}"}
        if not target.is_dir():
            return {"error": f"Not a directory: {path}"}

        entries = []
        for item in sorted(target.iterdir()):
            stat = item.stat()
            entries.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size_bytes": stat.st_size if item.is_file() else None,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
        return {"path": str(target.relative_to(ALLOWED_ROOT)), "entries": entries}
    except PermissionError as e:
        return {"error": str(e)}


def read_file(path: str, max_chars: int = 8000) -> dict:
    """Read the contents of a file (truncated if large)."""
    try:
        target = safe_path(path)
        if not target.exists():
            return {"error": f"File not found: {path}"}
        if not target.is_file():
            return {"error": f"Not a file: {path}"}

        content = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(content) > max_chars
        return {
            "path": path,
            "content": content[:max_chars],
            "truncated": truncated,
            "total_chars": len(content),
        }
    except PermissionError as e:
        return {"error": str(e)}


def write_file(path: str, content: str, mode: str = "overwrite") -> dict:
    """
    Write content to a file.
    mode: 'overwrite' (default) | 'append' | 'create_new' (fails if exists)
    """
    try:
        target = safe_path(path)
        if mode == "create_new" and target.exists():
            return {"error": f"File already exists: {path}. Use mode='overwrite' to replace it."}

        target.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        target.write_text(content, encoding="utf-8") if write_mode == "w" else \
            target.open("a", encoding="utf-8").write(content)

        return {"success": True, "path": path, "bytes_written": len(content.encode())}
    except PermissionError as e:
        return {"error": str(e)}


def search_files(query: str, search_in: str = ".", search_type: str = "both") -> dict:
    """
    Search files by name or content.
    search_type: 'name' | 'content' | 'both'
    """
    try:
        root = safe_path(search_in)
        results = []
        pattern = query.lower()

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(ALLOWED_ROOT))

            # Search by filename
            if search_type in ("name", "both"):
                if pattern in file_path.name.lower() or fnmatch.fnmatch(file_path.name.lower(), f"*{pattern}*"):
                    results.append({"path": rel, "match_type": "filename", "context": None})
                    continue

            # Search by content (text files only, skip large files)
            if search_type in ("content", "both") and file_path.stat().st_size < 1_000_000:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    if pattern in text.lower():
                        # Find context around first match
                        idx = text.lower().find(pattern)
                        start = max(0, idx - 60)
                        end = min(len(text), idx + 60)
                        context = "..." + text[start:end].strip() + "..."
                        results.append({"path": rel, "match_type": "content", "context": context})
                except Exception:
                    pass

        return {"query": query, "results": results[:50], "total_found": len(results)}
    except PermissionError as e:
        return {"error": str(e)}


def delete_file(path: str) -> dict:
    """Delete a file (not directories)."""
    try:
        target = safe_path(path)
        if not target.exists():
            return {"error": f"File not found: {path}"}
        if target.is_dir():
            return {"error": "Use a specific tool to delete directories. This tool only deletes files."}
        target.unlink()
        return {"success": True, "deleted": path}
    except PermissionError as e:
        return {"error": str(e)}


def get_file_info(path: str) -> dict:
    """Get metadata about a file or directory."""
    try:
        target = safe_path(path)
        if not target.exists():
            return {"error": f"Path not found: {path}"}
        stat = target.stat()
        return {
            "path": path,
            "type": "directory" if target.is_dir() else "file",
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "extension": target.suffix if target.is_file() else None,
        }
    except PermissionError as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# TOOL DEFINITIONS (sent to Claude)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "list_directory",
        "description": "List the files and subdirectories inside a folder. Use '.' for the root workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the directory (default: '.')"},
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read the text contents of a file. Returns up to 8000 characters by default.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "max_chars": {"type": "integer", "description": "Max characters to return (default 8000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or edit a file. Can overwrite, append, or create new files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "Text content to write"},
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append", "create_new"],
                    "description": "Write mode: overwrite (default), append, or create_new",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_files",
        "description": "Search for files by name or text content within the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "search_in": {"type": "string", "description": "Directory to search in (default: '.')"},
                "search_type": {
                    "type": "string",
                    "enum": ["name", "content", "both"],
                    "description": "Search by filename, file content, or both (default: 'both')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "delete_file",
        "description": "Permanently delete a file. Use with caution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file to delete"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_file_info",
        "description": "Get metadata about a file or directory (size, dates, type).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file or directory"},
            },
            "required": ["path"],
        },
    },
]


# ─────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────

TOOL_MAP = {
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "search_files": search_files,
    "delete_file": delete_file,
    "get_file_info": get_file_info,
}

def execute_tool(name: str, inputs: dict) -> str:
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_MAP[name](**inputs)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────
# AGENTIC LOOP
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a powerful local file system agent. You have access to a sandboxed workspace at:
  {ALLOWED_ROOT}

You can list, read, write, search, and delete files within this workspace.

Guidelines:
- Always use relative paths (e.g., 'reports/summary.txt', not absolute paths)
- Before editing a file, read it first to understand its current content
- For complex tasks, break them into steps and use multiple tools
- When summarizing files, be concise and highlight key points
- For workflows, explain each step as you go
- Never access paths outside the workspace — you don't have permission
- Confirm destructive actions (like deletes) by explaining what you're about to do

You are helpful, thorough, and precise."""


def run_agent(user_message: str):
    """Run the agentic loop for a user message."""
    print(f"\n{'='*60}")
    print(f"🧠 Agent thinking...")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]
    loop_count = 0

    while loop_count < MAX_AGENTIC_LOOPS:
        loop_count += 1

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect text output and tool calls
        tool_calls = []
        text_output = []

        for block in response.content:
            if block.type == "text":
                text_output.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        # Print any text the agent produced
        if text_output:
            print("\n🤖 Agent:", "\n".join(text_output))

        # If no tool calls, we're done
        if response.stop_reason == "end_turn" or not tool_calls:
            break

        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            print(f"\n🔧 Calling tool: {tool_call.name}")
            print(f"   Inputs: {json.dumps(tool_call.input, indent=2)}")

            result = execute_tool(tool_call.name, tool_call.input)
            result_data = json.loads(result)

            # Pretty print result summary
            if "error" in result_data:
                print(f"   ❌ Error: {result_data['error']}")
            elif "entries" in result_data:
                print(f"   ✅ Found {len(result_data['entries'])} items")
            elif "content" in result_data:
                print(f"   ✅ Read {result_data['total_chars']} chars (truncated: {result_data['truncated']})")
            elif "results" in result_data:
                print(f"   ✅ Search found {result_data['total_found']} matches")
            elif "success" in result_data:
                print(f"   ✅ Success")
            else:
                print(f"   ✅ Done")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result,
            })

        # Add assistant response + tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    if loop_count >= MAX_AGENTIC_LOOPS:
        print("\n⚠️  Max loops reached. Stopping agent.")

    print(f"\n{'='*60}")
    print(f"✅ Task complete ({loop_count} loop(s))")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════╗")
    print("║     🗂️  Local File System Agent           ║")
    print("║     Powered by Claude API                ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Workspace: {str(ALLOWED_ROOT)[:30]:<30} ║")
    print("╚══════════════════════════════════════════╝")
    print("\nType your task below. Examples:")
    print("  • List all files in the workspace")
    print("  • Summarize all .txt files")
    print("  • Search for files containing 'budget'")
    print("  • Create a report.md with a summary of all files")
    print("  • Find all TODO items across all files")
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            run_agent(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
