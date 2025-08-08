#!/usr/bin/env python3
"""
AI Suggestion Engine - Claude-powered code fix suggestions

This module provides AI-powered fix suggestions by combining ruff diagnostics
with Claude's analysis capabilities to offer intelligent, context-aware solutions.

Features:
- Context-aware fix suggestions using Claude
- Confidence scoring for suggestions
- Batch processing for multiple files
- Integration with ruff diagnostics
- Caching of suggestions for performance
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any

import aiofiles
from rich.console import Console


class FixConfidence(Enum):
    """Confidence levels for AI suggestions"""

    LOW = "low"  # < 0.4
    MEDIUM = "medium"  # 0.4 - 0.7
    HIGH = "high"  # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass
class SuggestionRequest:
    """Request for AI suggestion"""

    file_path: Path
    diagnostics: list[dict[str, Any]]
    code_context: str | None = None
    max_suggestions: int = 3
    include_explanation: bool = True
    confidence_threshold: float = 0.5


@dataclass
class SuggestionResponse:
    """AI suggestion response"""

    file_path: Path
    suggestions: list[dict[str, Any]]
    confidence: float
    explanation: str
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    claude_model: str = "haiku"


class AISuggestionEngine:
    """
    AI-powered suggestion engine using Claude for intelligent code fixes

    This engine analyzes ruff diagnostics and generates context-aware fix
    suggestions using Claude's code understanding capabilities.
    """

    def __init__(self, claude_cli: str = "claude", model: str = "haiku"):
        self.claude_cli = claude_cli
        self.model = model
        self.console = Console()
        self.logger = logging.getLogger("ai-suggestions")

        # Caching for performance
        self.suggestion_cache: dict[str, SuggestionResponse] = {}
        self.cache_ttl = timedelta(minutes=30)

        # Performance metrics
        self.stats = {
            "suggestions_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0,
        }

    async def generate_suggestions(self, request: SuggestionRequest) -> SuggestionResponse:
        """Generate AI-powered fix suggestions for diagnostics"""
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(request)
        cached_response = self._get_cached_response(cache_key)

        if cached_response:
            self.stats["cache_hits"] += 1
            return cached_response

        self.stats["cache_misses"] += 1

        try:
            # Prepare context for Claude
            context = await self._prepare_context(request)

            # Generate Claude prompt
            prompt = self._create_claude_prompt(request, context)

            # Call Claude
            claude_response = await self._call_claude(prompt)

            # Parse and structure response
            suggestions = self._parse_claude_response(claude_response, request)

            # Calculate confidence score
            confidence = self._calculate_confidence(suggestions, request.diagnostics)

            # Create response
            processing_time = time.time() - start_time
            response = SuggestionResponse(
                file_path=request.file_path,
                suggestions=suggestions,
                confidence=confidence,
                explanation=claude_response,
                processing_time=processing_time,
                claude_model=self.model,
            )

            # Cache the response
            self._cache_response(cache_key, response)

            # Update stats
            self.stats["suggestions_generated"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["suggestions_generated"]
            )

            return response

        except Exception as e:
            self.logger.exception(f"Error generating AI suggestions: {e}")
            # Return empty response on error
            return SuggestionResponse(
                file_path=request.file_path,
                suggestions=[],
                confidence=0.0,
                explanation=f"Error generating suggestions: {e}",
                processing_time=time.time() - start_time,
                claude_model=self.model,
            )

    async def _prepare_context(self, request: SuggestionRequest) -> str:
        """Prepare code context for Claude analysis"""
        if request.code_context:
            return request.code_context

        # Read file content if not provided
        try:
            async with aiofiles.open(request.file_path, encoding="utf-8") as f:
                content = await f.read()
            return content
        except Exception as e:
            self.logger.warning(f"Could not read file {request.file_path}: {e}")
            return ""

    def _create_claude_prompt(self, request: SuggestionRequest, context: str) -> str:
        """Create a focused prompt for Claude analysis"""

        # Extract diagnostic information
        diagnostic_summary = []
        for i, diag in enumerate(request.diagnostics[: request.max_suggestions], 1):
            diagnostic_summary.append(
                f"{i}. {diag.get('code', 'Unknown')}: {diag.get('message', 'No message')}"
            )
            if "location" in diag:
                loc = diag["location"]
                diagnostic_summary.append(
                    f"   Line {loc.get('row', '?')}, Column {loc.get('column', '?')}"
                )

        prompt = f"""Analyze this Python code and provide specific fix suggestions for ruff linter issues:

FILE: {request.file_path}

DIAGNOSTICS:
{chr(10).join(diagnostic_summary)}

CODE CONTEXT:
```python
{context[:2000]}  # Truncate for token limits
```

Please provide:
1. **Root Cause Analysis**: Brief explanation of each issue
2. **Specific Fix Suggestions**: Concrete code changes needed
3. **Priority Ranking**: Which issues to fix first
4. **Best Practices**: How to prevent similar issues

Format your response as JSON with this structure:
{{
    "suggestions": [
        {{
            "diagnostic_code": "E501",
            "title": "Fix line length violation",
            "fix_description": "Break long line into multiple lines",
            "confidence": 0.9,
            "priority": "high",
            "code_example": "# Fixed code here"
        }}
    ],
    "summary": "Overall analysis and recommendations"
}}

Keep suggestions practical and specific to the actual code shown."""

        return prompt

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude CLI with the prepared prompt"""
        try:
            # Write prompt to temporary file to handle large prompts
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(prompt)
                prompt_file = f.name

            # Call Claude with timeout
            process = await asyncio.create_subprocess_exec(
                self.claude_cli,
                "--model",
                self.model,
                "--file",
                prompt_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45.0)

            # Clean up temp file
            Path(prompt_file).unlink(missing_ok=True)

            if process.returncode != 0:
                raise RuntimeError(f"Claude CLI error: {stderr.decode()}")

            return stdout.decode().strip()

        except TimeoutError:
            raise RuntimeError("Claude API call timed out")
        except Exception as e:
            raise RuntimeError(f"Error calling Claude: {e}")

    def _parse_claude_response(
        self, response: str, request: SuggestionRequest
    ) -> list[dict[str, Any]]:
        """Parse Claude's JSON response into structured suggestions"""
        try:
            # Try to extract JSON from response
            response_clean = response.strip()

            # Look for JSON block in the response
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_clean[json_start:json_end]
                parsed = json.loads(json_str)

                suggestions = parsed.get("suggestions", [])

                # Validate and enrich suggestions
                validated_suggestions = []
                for suggestion in suggestions:
                    if self._validate_suggestion(suggestion):
                        validated_suggestions.append(suggestion)

                return validated_suggestions[: request.max_suggestions]

            # Fallback: create simple suggestion from raw response
            return [
                {
                    "title": "AI Analysis",
                    "fix_description": response[:500],  # Truncate
                    "confidence": 0.5,
                    "priority": "medium",
                }
            ]

        except json.JSONDecodeError:
            # If JSON parsing fails, create a simple suggestion
            return [
                {
                    "title": "AI Suggestion",
                    "fix_description": response[:500],
                    "confidence": 0.3,
                    "priority": "low",
                }
            ]

    def _validate_suggestion(self, suggestion: dict[str, Any]) -> bool:
        """Validate that a suggestion has required fields"""
        required_fields = ["title", "fix_description"]
        return all(field in suggestion for field in required_fields)

    def _calculate_confidence(
        self, suggestions: list[dict[str, Any]], diagnostics: list[dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score for suggestions"""
        if not suggestions:
            return 0.0

        # Base confidence on individual suggestion confidences
        total_confidence = sum(suggestion.get("confidence", 0.5) for suggestion in suggestions)

        # Adjust based on number of diagnostics vs suggestions
        diagnostic_coverage = min(len(suggestions) / max(len(diagnostics), 1), 1.0)

        # Calculate weighted average
        avg_confidence = total_confidence / len(suggestions)

        # Apply coverage penalty
        final_confidence = avg_confidence * (0.7 + 0.3 * diagnostic_coverage)

        return min(final_confidence, 1.0)

    def _get_cache_key(self, request: SuggestionRequest) -> str:
        """Generate cache key for request"""
        # Create hash based on file path and diagnostic codes
        diagnostic_codes = [diag.get("code", "") for diag in request.diagnostics]
        key_data = f"{request.file_path}:{':'.join(sorted(diagnostic_codes))}"
        return str(hash(key_data))

    def _get_cached_response(self, cache_key: str) -> SuggestionResponse | None:
        """Get cached response if still valid"""
        if cache_key in self.suggestion_cache:
            cached = self.suggestion_cache[cache_key]
            if datetime.now() - cached.timestamp < self.cache_ttl:
                return cached
            else:
                # Remove expired cache entry
                del self.suggestion_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: SuggestionResponse) -> None:
        """Cache the response"""
        self.suggestion_cache[cache_key] = response

        # Limit cache size
        if len(self.suggestion_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.suggestion_cache.keys(),
                key=lambda k: self.suggestion_cache[k].timestamp,
            )[:20]
            for key in oldest_keys:
                del self.suggestion_cache[key]

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            "cache_size": len(self.suggestion_cache),
            "cache_hit_rate": (
                self.stats["cache_hits"]
                / max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
            ),
        }

    def clear_cache(self) -> None:
        """Clear suggestion cache"""
        self.suggestion_cache.clear()
        self.logger.info("AI suggestion cache cleared")
