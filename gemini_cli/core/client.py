"""Standalone Gemini client with business service account authentication only."""

import logging
from typing import Any

import vertexai
from vertexai.generative_models import FunctionDeclaration
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Tool

from .auth import GeminiAuth

logger = logging.getLogger(__name__)


class GeminiClient:
    """Simplified Gemini client for CLI usage."""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp") -> None:
        """Initialize the Gemini client.

        Args:
            model_name: The Gemini model to use
        """
        self.model_name = model_name
        self.model: GenerativeModel | None = None
        self._authenticated = False

    def _ensure_authenticated(self) -> None:
        """Ensure authentication and model initialization."""
        if self._authenticated:
            return

        try:
            # Setup authentication
            GeminiAuth.setup_environment()
            credentials, project_id = GeminiAuth.get_credentials()

            # Initialize Vertex AI
            vertexai.init(
                project=project_id,
                location=GeminiAuth.LOCATION,
                credentials=credentials,
            )

            # Initialize model WITHOUT tools initially - tools will be added per request
            # CRITICAL FIX: Keep base model without tools to avoid conflicts
            self.model = GenerativeModel(
                self.model_name,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=4096,
                ),
                # NOTE: No tools here - they are added dynamically in process_with_tools()
            )

            self._authenticated = True
            logger.info(f"✅ Authenticated with model: {self.model_name}")

        except Exception:
            logger.exception("❌ Authentication failed")
            raise

    async def process(self, prompt: str) -> str:
        """Process a single prompt and return response.

        Args:
            prompt: The user prompt

        Returns:
            The model's response
        """
        self._ensure_authenticated()

        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            response = self.model.generate_content(prompt)
            return str(response.text)
        except Exception:
            logger.exception("Generation failed")
            raise

    async def process_with_tools(self, prompt: str, tools: list[Any]) -> str:
        """Process prompt with tools (for ReAct engine).

        Args:
            prompt: The user prompt
            tools: List of available tools (Tool instances from our registry)

        Returns:
            The model's response
        """
        self._ensure_authenticated()

        try:
            # Convert our tool registry tools to Vertex AI function declarations
            function_declarations = []

            for tool in tools:
                # Create function declaration for each tool
                func_decl = FunctionDeclaration(
                    name=tool.name.replace("-", "_").replace(
                        " ", "_"
                    ),  # Ensure valid function name
                    description=tool.description,
                    parameters={
                        "type": "object",
                        "properties": self._get_tool_parameters(tool),
                        "required": self._get_required_parameters(tool),
                    },
                )
                function_declarations.append(func_decl)

            # Create Vertex AI Tool with function declarations
            vertex_tool = (
                Tool(function_declarations=function_declarations) if function_declarations else None
            )

            # Create model with tools
            model_with_tools = GenerativeModel(
                self.model_name,
                tools=[vertex_tool] if vertex_tool else None,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=4096,
                ),
            )

            response = model_with_tools.generate_content(prompt)

            # Handle function calls if present
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate.content, "parts") and candidate.content.parts:
                    function_results = []
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            call = part.function_call
                            if call and hasattr(call, "name"):
                                logger.info(
                                    f"Function called: {call.name} with args: {dict(call.args) if hasattr(call, 'args') else {}}"
                                )
                                function_results.append(
                                    {
                                        "function": call.name,
                                        "args": dict(call.args) if hasattr(call, "args") else {},
                                        "result": "Function execution would happen here",
                                    }
                                )

                    if function_results:
                        # Return function call information
                        return f"Function calls made: {function_results}\n\n✅ Function calling is ENABLED!"

            # Try to get text response
            try:
                return str(response.text) if response.text else "No response generated"
            except (AttributeError, ValueError):
                # No text response when function is called
                return "Function call completed (no text response)"
        except Exception:
            logger.exception("Tool-assisted generation failed")
            raise

    def _get_tool_parameters(self, tool: Any) -> dict[str, Any]:
        """Extract parameter schema from tool.

        This is a basic implementation - tools should ideally provide their own schema.
        """
        # Basic parameter extraction based on common tool patterns
        if hasattr(tool, "get_parameters"):
            return tool.get_parameters()

        # Default parameters for common tool actions
        if tool.name == "filesystem":
            return {
                "action": {
                    "type": "string",
                    "description": "Action to perform: read_file, write_file, list_files, search_files, search_content",
                },
                "path": {"type": "string", "description": "File or directory path"},
                "content": {"type": "string", "description": "Content for write operations"},
                "pattern": {"type": "string", "description": "Pattern for search operations"},
            }
        elif tool.name == "code_analysis":
            return {
                "action": {
                    "type": "string",
                    "description": "Analysis action: analyze_complexity, check_style, find_issues",
                },
                "path": {"type": "string", "description": "Path to analyze"},
                "language": {"type": "string", "description": "Programming language"},
            }

        # Generic fallback
        return {
            "action": {"type": "string", "description": "Action to perform"},
            "params": {"type": "object", "description": "Additional parameters"},
        }

    def _get_required_parameters(self, tool: Any) -> list[str]:
        """Get required parameters for a tool."""
        if hasattr(tool, "get_required_parameters"):
            return tool.get_required_parameters()

        # Default required parameters
        if tool.name in ["filesystem", "code_analysis"]:
            return ["action"]

        return []
