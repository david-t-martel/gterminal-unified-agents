#!/usr/bin/env python3
"""
Enhanced debugging and development tools integration for gterminal.
Integrates rufft-claude.sh with file watchers and external tools.
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class ToolConfig:
    """Configuration for external debugging tools."""

    name: str
    command: list[str]
    file_patterns: set[str]
    auto_trigger: bool = True
    timeout: int = 30
    priority: int = 5


class EnhancedDebugToolchain:
    """Enhanced debugging toolchain for gterminal integration."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.scripts_dir = self.project_root / "scripts"
        self.rufft_claude_path = self.scripts_dir / "rufft-claude.sh"
        self.tool_results: dict[str, Any] = {}
        self.setup_tools()

    def setup_tools(self) -> None:
        """Configure integrated debugging tools."""
        self.tools = {
            "ruff": ToolConfig(
                name="ruff",
                command=["uv", "run", "ruff", "check"],
                file_patterns={".py"},
                auto_trigger=True,
                priority=8,
            ),
            "mypy": ToolConfig(
                name="mypy",
                command=["uv", "run", "mypy"],
                file_patterns={".py"},
                auto_trigger=True,
                priority=7,
            ),
            "rufft_claude_auto": ToolConfig(
                name="rufft_claude_auto",
                command=["uv", "run", str(self.rufft_claude_path), "auto-fix"],
                file_patterns={".py"},
                auto_trigger=True,
                priority=10,
            ),
            "rufft_claude_ai": ToolConfig(
                name="rufft_claude_ai",
                command=["uv", "run", str(self.rufft_claude_path), "ai-suggest"],
                file_patterns={".py"},
                auto_trigger=False,
                priority=9,
            ),
        }

    async def run_tool(self, tool_name: str, target_path: Path) -> dict[str, Any]:
        """Run a specific tool on a target path."""
        if tool_name not in self.tools:
            console.print(f"❌ Unknown tool: {tool_name}", style="red")
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        tool_config = self.tools[tool_name]
        console.print(f"🔧 Running {tool_name} on {target_path}", style="blue")

        try:
            cmd = tool_config.command.copy()
            cmd.append(str(target_path))

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=tool_config.timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return {"success": False, "error": f"Tool {tool_name} timed out"}

            result = {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "tool": tool_name,
                "target": str(target_path),
            }

            self.tool_results[f"{tool_name}_{target_path}"] = result

            if result["success"]:
                console.print(f"✅ {tool_name} completed successfully", style="green")
            else:
                console.print(f"❌ {tool_name} failed", style="red")
                if result["stderr"]:
                    stderr_str = str(result["stderr"])
                    console.print(f"Error: {stderr_str[:200]}...", style="red")

            return result

        except (OSError, subprocess.SubprocessError) as e:
            error_result = {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "target": str(target_path),
            }
            console.print(f"❌ Error running {tool_name}: {e}", style="red")
            return error_result

    async def run_tool_suite(
        self, target_path: Path, tool_filter: str | None = None
    ) -> dict[str, Any]:
        """Run a suite of tools on a target path."""
        console.print(f"🧪 Running tool suite on {target_path}", style="bold blue")

        tools_to_run = list(self.tools.items())
        if tool_filter:
            tools_to_run = [
                (name, config)
                for name, config in tools_to_run
                if tool_filter.lower() in name.lower()
            ]

        tools_to_run = sorted(tools_to_run, key=lambda x: x[1].priority, reverse=True)

        results = {}
        for tool_name, tool_config in tools_to_run:
            if target_path.suffix in tool_config.file_patterns:
                result = await self.run_tool(tool_name, target_path)
                results[tool_name] = result

        return results

    def create_tools_table(self) -> Table:
        """Create table showing tool status."""
        table = Table(title="Tool Status", show_header=True, header_style="bold magenta")
        table.add_column("Tool")
        table.add_column("Status")
        table.add_column("Auto-Trigger")
        table.add_column("Priority")

        for tool_name, tool_config in self.tools.items():
            status = "🟢 Ready"
            auto_trigger = "✅" if tool_config.auto_trigger else "❌"
            priority = str(tool_config.priority)

            table.add_row(tool_name, status, auto_trigger, priority)

        return table

    async def monitor_mode(self) -> None:
        """Start monitoring mode with real-time display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        with Live(layout, refresh_per_second=1, screen=True):
            while True:
                layout["header"].update(
                    Panel("🔧 Enhanced Debug Toolchain - GTerminal", style="bold blue")
                )

                layout["main"].update(self.create_tools_table())

                layout["footer"].update(
                    Panel(
                        "Enhanced debugging tools ready | Press Ctrl+C to exit",
                        style="dim",
                    )
                )

                await asyncio.sleep(1)


async def main() -> None:
    """Main entry point for the enhanced debug toolchain."""
    parser = argparse.ArgumentParser(description="Enhanced Debug Toolchain for GTerminal")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")
    parser.add_argument("--run-suite", help="Run tool suite on file/directory")
    parser.add_argument("--tool-filter", help="Filter tools by name")

    args = parser.parse_args()

    toolchain = EnhancedDebugToolchain()

    if args.monitor:
        try:
            await toolchain.monitor_mode()
        except KeyboardInterrupt:
            console.print("\n🛑 Shutting down toolchain...", style="yellow")
            console.print("✅ Toolchain stopped", style="green")

    elif args.run_suite:
        target_path = Path(args.run_suite)
        if not target_path.exists():
            console.print(f"❌ Path not found: {target_path}", style="red")
            sys.exit(1)

        results = await toolchain.run_tool_suite(target_path, args.tool_filter)

        console.print("\n📊 Tool Suite Results:", style="bold blue")
        for tool_name, result in results.items():
            status = "✅" if result.get("success") else "❌"
            console.print(f"  {status} {tool_name}")
    else:
        console.print("Enhanced Debug Toolchain for GTerminal", style="bold blue")
        console.print("\nUsage:", style="yellow")
        console.print("  --monitor          Start monitoring mode")
        console.print("  --run-suite FILE   Run tool suite on file/directory")
        console.print("  --tool-filter STR  Filter tools by name")
        console.print("\nExample:", style="cyan")
        console.print("  uv run python enhanced_toolchain.py --run-suite myfile.py")


if __name__ == "__main__":
    asyncio.run(main())
