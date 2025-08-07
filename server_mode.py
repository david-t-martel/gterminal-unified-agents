#!/usr/bin/env python3
"""Background Server Mode for gterminal ReAct Agent.

This creates a background server that can run autonomously and accept
function calls via MCP or HTTP API for continuous processing.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Any

from gemini_cli.core.react_engine import SimpleReactEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GTerminalReactServer:
    """Background server for gterminal ReAct agent."""

    def __init__(self, port: int = 8765):
        """Initialize the server.

        Args:
            port: Port to run the server on
        """
        self.port = port
        self.react_engine = SimpleReactEngine()
        self.is_running = False
        self.session_id = f"gterminal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.task_queue = asyncio.Queue()
        self.results_store: dict[str, Any] = {}

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"🚀 GTerminal ReAct Server initialized - Session: {self.session_id}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
        sys.exit(0)

    async def process_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Process a request using the ReAct engine.

        Args:
            request_data: Request data containing prompt and optional parameters

        Returns:
            Response data with results
        """
        request_id = request_data.get("id", f"req_{datetime.now().isoformat()}")
        prompt = request_data.get("prompt", request_data.get("message", ""))

        if not prompt:
            return {
                "id": request_id,
                "error": "No prompt provided",
                "timestamp": datetime.now().isoformat(),
            }

        logger.info(f"🔄 Processing request {request_id}: {prompt[:100]}...")

        try:
            # Use ReAct engine to process the request
            result = await self.react_engine.process(prompt)

            # Get execution summary
            summary = self.react_engine.get_execution_summary()

            response = {
                "id": request_id,
                "result": result,
                "execution_summary": summary,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "status": "completed",
            }

            # Store result
            self.results_store[request_id] = response

            logger.info(f"✅ Completed request {request_id}")
            return response

        except Exception as e:
            error_response = {
                "id": request_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "status": "error",
            }

            logger.error(f"❌ Request {request_id} failed: {e}")
            return error_response

    async def background_worker(self):
        """Background worker to process queued tasks."""
        logger.info("🔄 Background worker started")

        while self.is_running:
            try:
                # Wait for tasks with timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Process the task
                result = await self.process_request(task)
                logger.info(f"Background task completed: {result.get('id')}")

                # Mark task as done
                self.task_queue.task_done()

            except TimeoutError:
                # No tasks available, continue loop
                continue
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                continue

    async def queue_task(self, request_data: dict[str, Any]) -> str:
        """Queue a task for background processing.

        Args:
            request_data: Request data

        Returns:
            Task ID for tracking
        """
        task_id = f"task_{datetime.now().isoformat()}"
        request_data["id"] = task_id

        await self.task_queue.put(request_data)
        logger.info(f"📝 Queued background task: {task_id}")

        return task_id

    async def get_result(self, request_id: str) -> dict[str, Any] | None:
        """Get result for a completed request.

        Args:
            request_id: Request ID to lookup

        Returns:
            Result data if available
        """
        return self.results_store.get(request_id)

    async def start_server(self):
        """Start the background server."""
        logger.info(f"🚀 Starting GTerminal ReAct Server on port {self.port}")
        self.is_running = True

        # Start background worker
        worker_task = asyncio.create_task(self.background_worker())

        # Start HTTP server (simple implementation)
        await self._start_http_server()

        # Wait for worker
        await worker_task

    async def _start_http_server(self):
        """Start simple HTTP server for API access."""
        from aiohttp import web, web_request

        async def handle_process(request: web_request.Request) -> web.Response:
            """Handle process request."""
            try:
                data = await request.json()
                result = await self.process_request(data)
                return web.json_response(result)
            except Exception as e:
                return web.json_response(
                    {"error": str(e), "timestamp": datetime.now().isoformat()}, status=500
                )

        async def handle_queue(request: web_request.Request) -> web.Response:
            """Handle queue request."""
            try:
                data = await request.json()
                task_id = await self.queue_task(data)
                return web.json_response(
                    {
                        "task_id": task_id,
                        "status": "queued",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                return web.json_response(
                    {"error": str(e), "timestamp": datetime.now().isoformat()}, status=500
                )

        async def handle_result(request: web_request.Request) -> web.Response:
            """Handle result lookup."""
            request_id = request.match_info.get("request_id")
            result = await self.get_result(request_id)

            if result:
                return web.json_response(result)
            else:
                return web.json_response(
                    {"error": "Result not found", "request_id": request_id}, status=404
                )

        async def handle_status(request: web_request.Request) -> web.Response:
            """Handle status request."""
            return web.json_response(
                {
                    "status": "running" if self.is_running else "stopped",
                    "session_id": self.session_id,
                    "queued_tasks": self.task_queue.qsize(),
                    "completed_tasks": len(self.results_store),
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                }
            )

        # Setup routes
        app = web.Application()
        app.router.add_post("/process", handle_process)
        app.router.add_post("/queue", handle_queue)
        app.router.add_get("/result/{request_id}", handle_result)
        app.router.add_get("/status", handle_status)
        app.router.add_get("/", handle_status)  # Root route

        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()

        logger.info(f"✅ HTTP API server started on http://localhost:{self.port}")
        logger.info("📍 Endpoints:")
        logger.info("   POST /process - Process request immediately")
        logger.info("   POST /queue   - Queue request for background processing")
        logger.info("   GET  /result/{id} - Get result by request ID")
        logger.info("   GET  /status  - Server status and metrics")

        # Keep server running
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            await runner.cleanup()


async def main():
    """Main entry point for server mode."""
    import argparse

    parser = argparse.ArgumentParser(description="GTerminal ReAct Background Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to run server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("🎯 STARTING GTERMINAL REACT BACKGROUND SERVER")
    logger.info("=" * 60)

    server = GTerminalReactServer(port=args.port)

    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
