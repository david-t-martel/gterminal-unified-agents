from typing import Any

#!/usr/bin/env python3
"""
Terminal Monitor for Universal Gemini Project Orchestrator
Provides real-time monitoring of autonomous agent tasks.
"""

from datetime import datetime
import sys
import time

import requests


class OrchestratorMonitor:
    def __init__(self, orchestrator_url: str = "http://127.0.0.1:8100") -> None:
        self.orchestrator_url = orchestrator_url
        self.last_task_count = 0
        self.monitoring = True

    def clear_screen(self) -> None:
        """Clear terminal screen."""

    def print_header(self) -> None:
        """Print monitoring header."""

    def print_health_status(self) -> None:
        """Print health status."""
        try:
            response = requests.get(f"{self.orchestrator_url}/health", timeout=5)
            health = response.json()

            "✅" if health["status"] == "healthy" else "❌"
            "🤖" if health.get("gemini_enabled") else "🚫"

        except Exception:
            pass

    def print_task_status(self) -> None:
        """Print current task status."""
        try:
            response = requests.get(f"{self.orchestrator_url}/tasks/status", timeout=5)
            status = response.json()

            status.get("total", 0)
            status.get("running", 0)
            status.get("completed", 0)
            status.get("failed", 0)

            # Show active tasks
            if status.get("active_tasks"):
                for task_info in status["active_tasks"].values():
                    if task_info["status"] == "running":
                        task_info["agent_type"]
                        task_info["task"][:50] + "..." if len(
                            task_info["task"]
                        ) > 50 else task_info["task"]
                        self.calculate_elapsed(task_info["started"])

                        # Show progress if available
                        if task_info.get("progress"):
                            task_info["progress"][-1]

        except Exception:
            pass

    def calculate_elapsed(self, started_time: str) -> str:
        """Calculate elapsed time."""
        try:
            start = datetime.fromisoformat(started_time)
            elapsed = datetime.now() - start

            total_seconds = int(elapsed.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60

            return f"{minutes}m {seconds}s"
        except:
            return "unknown"

    def print_recent_completions(self) -> None:
        """Print recently completed tasks."""
        try:
            response = requests.get(f"{self.orchestrator_url}/tasks/status", timeout=5)
            status = response.json()

            completed_tasks: list[Any] = []
            for task_id, task_info in status.get("active_tasks", {}).items():
                if task_info["status"] == "completed":
                    completed_tasks.append((task_id, task_info))

            if completed_tasks:
                # Show last 3 completed tasks
                for task_id, task_info in completed_tasks[-3:]:
                    task_info["agent_type"]
                    completed_time = task_info.get("completed", "unknown")

                    try:
                        completed_dt = datetime.fromisoformat(completed_time)
                        completed_dt.strftime("%H:%M:%S")
                    except:
                        pass

                    # Show result summary if available
                    if task_info.get("result"):
                        result = task_info["result"]
                        if isinstance(result, dict):
                            if "files_processed" in result:
                                pass
                            if "actions" in result and isinstance(result["actions"], list):
                                pass

        except Exception:
            pass

    def print_instructions(self) -> None:
        """Print usage instructions."""

    def run_monitor(self) -> None:
        """Run the monitoring loop."""
        try:
            while self.monitoring:
                self.clear_screen()
                self.print_header()
                self.print_health_status()
                self.print_task_status()
                self.print_recent_completions()
                self.print_instructions()

                # Update status line

                time.sleep(10)

        except KeyboardInterrupt:
            pass
        except Exception:
            pass


def main() -> None:
    """Main function."""
    orchestrator_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"

    monitor = OrchestratorMonitor(orchestrator_url)
    monitor.run_monitor()


if __name__ == "__main__":
    main()
