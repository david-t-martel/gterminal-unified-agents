"""Standalone Gemini client with business service account authentication only."""

import logging
from typing import Any

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

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
        self.model = None
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
                credentials=credentials
            )
            
            # Initialize model
            self.model = GenerativeModel(
                self.model_name,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=4096,
                )
            )
            
            self._authenticated = True
            logger.info(f"✅ Authenticated with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
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
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def process_with_tools(self, prompt: str, tools: list[Any]) -> str:
        """Process prompt with tools (for ReAct engine).
        
        Args:
            prompt: The user prompt
            tools: List of available tools
            
        Returns:
            The model's response
        """
        self._ensure_authenticated()
        
        try:
            # Create model with tools
            model_with_tools = GenerativeModel(
                self.model_name,
                tools=tools,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=4096,
                )
            )
            
            response = model_with_tools.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Tool-assisted generation failed: {e}")
            raise