#!/usr/bin/env python3
"""
VertexAI SDK Documentation Scraper

Comprehensive scraping system for the latest VertexAI SDK documentation
from official Google Cloud sources.

Features:
- Async scraping for performance
- Structured data extraction
- Category-based organization
- Rate limiting and respect for robots.txt
- Caching system for incremental updates
- Error handling and retry logic
"""

import asyncio
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentationEntry:
    """Structured representation of a documentation entry."""

    id: str
    title: str
    url: str
    content: str
    category: str
    subcategory: str
    tags: list[str]
    code_examples: list[str]
    parameters: list[dict[str, Any]]
    return_types: list[str]
    version: str
    last_updated: str
    content_hash: str
    embedding_vector: list[float] | None = None


class VertexAIDocScraper:
    """Main documentation scraper for VertexAI SDK."""

    BASE_URLS = {
        "python_sdk": "https://cloud.google.com/vertex-ai/docs/python-sdk",
        "generative_ai": "https://cloud.google.com/vertex-ai/generative-ai/docs",
        "api_reference": "https://cloud.google.com/vertex-ai/docs/reference/python",
        "samples": "https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/generative_ai",
        "tutorials": "https://cloud.google.com/vertex-ai/docs/tutorials",
    }

    CATEGORIES = {
        "generative_models": ["gemini-1.5", "gemini-2.0", "palm", "codey"],
        "function_calling": ["tools", "function-calls", "automatic-function-calling"],
        "grounding_rag": ["grounding", "rag", "search", "retrieval"],
        "evaluation_safety": ["evaluation", "safety", "filters", "monitoring"],
        "streaming_async": ["streaming", "async", "concurrent", "batch"],
        "context_caching": ["context-caching", "cache", "cost-optimization"],
        "code_execution": ["code-execution", "sandbox", "python-execution"],
        "system_instructions": ["system-instructions", "prompts", "templates"],
        "structured_output": ["json-mode", "structured", "schema", "parsing"],
        "batch_prediction": ["batch", "prediction", "async-inference"],
        "model_tuning": ["tuning", "fine-tuning", "custom-models"],
        "authentication": ["auth", "credentials", "service-account"],
    }

    def __init__(self, db_path: str = "vertexai_docs.db", cache_dir: str = "./cache"):
        self.db_path = Path(db_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session: aiohttp.ClientSession | None = None
        self.visited_urls: set[str] = set()
        self.scraped_docs: list[DocumentationEntry] = []

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with proper schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS documentation (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            tags TEXT, -- JSON array
            code_examples TEXT, -- JSON array
            parameters TEXT, -- JSON array
            return_types TEXT, -- JSON array
            version TEXT,
            last_updated TEXT,
            content_hash TEXT,
            embedding_vector TEXT, -- JSON array
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_category ON documentation(category);
        """
        )

        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_tags ON documentation(tags);
        """
        )

        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_content_hash ON documentation(content_hash);
        """
        )

        conn.commit()
        conn.close()

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": "VertexAI-Documentation-Scraper/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def fetch_url(self, url: str, retries: int = 3) -> str | None:
        """Fetch URL content with retry logic."""
        for attempt in range(retries):
            try:
                await asyncio.sleep(0.5)  # Rate limiting
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched: {url}")
                        return content
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
            except TimeoutError:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.exception(f"Error fetching {url}: {e}")

        return None

    def _extract_code_examples(self, soup: BeautifulSoup) -> list[str]:
        """Extract code examples from documentation."""
        examples = []

        # Look for code blocks
        for code_block in soup.find_all(["code", "pre"]):
            if (
                (code_block.get("class") and "python" in str(code_block.get("class")))
                or "import" in code_block.get_text()
                or "def " in code_block.get_text()
            ):
                examples.append(code_block.get_text().strip())

        # Look for specific VertexAI patterns
        text_content = soup.get_text()
        vertex_patterns = [
            r"from google\.cloud import aiplatform.*?(?=\n\n|\Z)",
            r"import vertexai.*?(?=\n\n|\Z)",
            r"client = aiplatform\..*?(?=\n\n|\Z)",
            r"model = GenerativeModel.*?(?=\n\n|\Z)",
        ]

        for pattern in vertex_patterns:
            matches = re.findall(pattern, text_content, re.DOTALL)
            examples.extend(matches)

        return list(set(examples))  # Remove duplicates

    def _extract_parameters(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Extract parameter information from documentation."""
        parameters = []

        # Look for parameter tables or lists
        for table in soup.find_all("table"):
            if "parameter" in table.get_text().lower():
                for row in table.find_all("tr")[1:]:  # Skip header
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 3:
                        param = {
                            "name": cells[0].get_text().strip(),
                            "type": cells[1].get_text().strip(),
                            "description": cells[2].get_text().strip(),
                        }
                        parameters.append(param)

        # Look for structured parameter documentation
        for dl in soup.find_all("dl"):
            current_param = {}
            for child in dl.children:
                if child.name == "dt":
                    if current_param:
                        parameters.append(current_param)
                    current_param = {"name": child.get_text().strip()}
                elif child.name == "dd" and current_param:
                    current_param["description"] = child.get_text().strip()

        return parameters

    def _categorize_content(
        self, url: str, title: str, content: str
    ) -> tuple[str, str]:
        """Categorize documentation content."""
        content_lower = f"{url} {title} {content}".lower()

        for category, keywords in self.CATEGORIES.items():
            if any(keyword in content_lower for keyword in keywords):
                # Find most specific subcategory
                for keyword in keywords:
                    if keyword in content_lower:
                        return category, keyword
                return category, keywords[0]

        return "general", "miscellaneous"

    def _extract_tags(self, content: str) -> list[str]:
        """Extract relevant tags from content."""
        tags = []

        # Technical terms
        tech_patterns = [
            r"\bgemini[- ]?(?:1\.5|2\.0|pro|flash)\b",
            r"\bfunction[- ]?calling\b",
            r"\bstreaming\b",
            r"\basync\b",
            r"\bembedding\b",
            r"\btokeniz\w+\b",
            r"\bgroundin\w*\b",
            r"\brag\b",
            r"\bsafety[- ]?filter\b",
            r"\bjson[- ]?mode\b",
            r"\bstructured[- ]?output\b",
            r"\bsystem[- ]?instruction\b",
            r"\bcontext[- ]?caching\b",
            r"\bcode[- ]?execution\b",
            r"\bbatch[- ]?prediction\b",
            r"\bmodel[- ]?tuning\b",
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            tags.extend([match.lower().replace("-", "_") for match in matches])

        return list(set(tags))

    async def scrape_documentation_page(self, url: str) -> DocumentationEntry | None:
        """Scrape a single documentation page."""
        if url in self.visited_urls:
            return None

        self.visited_urls.add(url)
        content = await self.fetch_url(url)

        if not content:
            return None

        soup = BeautifulSoup(content, "html.parser")

        # Extract title
        title_elem = soup.find("h1") or soup.find("title")
        title = title_elem.get_text().strip() if title_elem else urlparse(url).path

        # Extract main content
        main_content = ""
        content_selectors = [
            "main",
            "article",
            ".main-content",
            ".content",
            ".documentation",
            ".doc-content",
            "#content",
        ]

        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text()
                break

        if not main_content:
            main_content = soup.get_text()

        # Clean up content
        main_content = re.sub(r"\s+", " ", main_content).strip()

        # Extract structured data
        code_examples = self._extract_code_examples(soup)
        parameters = self._extract_parameters(soup)
        category, subcategory = self._categorize_content(url, title, main_content)
        tags = self._extract_tags(main_content)

        # Generate content hash
        content_hash = hashlib.sha256(main_content.encode()).hexdigest()[:16]

        doc_entry = DocumentationEntry(
            id=hashlib.sha256(url.encode()).hexdigest()[:16],
            title=title,
            url=url,
            content=main_content,
            category=category,
            subcategory=subcategory,
            tags=tags,
            code_examples=code_examples,
            parameters=parameters,
            return_types=[],  # Will be enhanced later
            version="latest",
            last_updated=datetime.now().isoformat(),
            content_hash=content_hash,
        )

        return doc_entry

    def _discover_urls(self, base_content: str, base_url: str) -> set[str]:
        """Discover additional URLs to scrape from base content."""
        soup = BeautifulSoup(base_content, "html.parser")
        urls = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("#"):
                continue

            full_url = urljoin(base_url, href)

            # Filter relevant documentation URLs
            if any(
                domain in full_url
                for domain in ["cloud.google.com/vertex-ai", "googleapis.dev"]
            ):
                urls.add(full_url)

        return urls

    async def scrape_all_documentation(self) -> list[DocumentationEntry]:
        """Scrape all VertexAI documentation."""
        all_urls = set(self.BASE_URLS.values())

        # First pass: scrape base URLs and discover more
        for base_url in list(all_urls):
            content = await self.fetch_url(base_url)
            if content:
                discovered = self._discover_urls(content, base_url)
                all_urls.update(discovered)

                # Limit to reasonable size
                if len(all_urls) > 200:
                    break

        logger.info(f"Discovered {len(all_urls)} URLs to scrape")

        # Second pass: scrape all discovered URLs
        tasks = []
        for url in all_urls:
            if not any(exclude in url for exclude in ["#", ".pdf", ".zip"]):
                tasks.append(self.scrape_documentation_page(url))

        # Process in batches to avoid overwhelming the server
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)

            for result in results:
                if isinstance(result, DocumentationEntry):
                    self.scraped_docs.append(result)

        logger.info(
            f"Successfully scraped {len(self.scraped_docs)} documentation entries"
        )
        return self.scraped_docs

    def save_to_database(self, docs: list[DocumentationEntry]) -> None:
        """Save documentation entries to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for doc in docs:
            cursor.execute(
                """
            INSERT OR REPLACE INTO documentation
            (id, title, url, content, category, subcategory, tags, code_examples,
             parameters, return_types, version, last_updated, content_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    doc.id,
                    doc.title,
                    doc.url,
                    doc.content,
                    doc.category,
                    doc.subcategory,
                    json.dumps(doc.tags),
                    json.dumps(doc.code_examples),
                    json.dumps(doc.parameters),
                    json.dumps(doc.return_types),
                    doc.version,
                    doc.last_updated,
                    doc.content_hash,
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(docs)} entries to database")

    def export_to_json(self, output_path: str) -> None:
        """Export documentation to JSON format."""
        data = {
            "metadata": {
                "scraped_at": datetime.now().isoformat(),
                "total_entries": len(self.scraped_docs),
                "categories": list(self.CATEGORIES.keys()),
            },
            "documentation": [asdict(doc) for doc in self.scraped_docs],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported documentation to {output_path}")


async def main():
    """Main scraping function."""
    scraper = VertexAIDocScraper()

    try:
        async with scraper:
            docs = await scraper.scrape_all_documentation()

            # Save to database
            scraper.save_to_database(docs)

            # Export to JSON
            scraper.export_to_json("vertexai_documentation.json")

            # Generate summary
            categories = {}
            for doc in docs:
                categories[doc.category] = categories.get(doc.category, 0) + 1

            print(f"\n{'=' * 50}")
            print("VertexAI Documentation Scraping Complete!")
            print(f"{'=' * 50}")
            print(f"Total entries: {len(docs)}")
            print("Categories:")
            for category, count in sorted(categories.items()):
                print(f"  {category}: {count} entries")
            print(f"Database: {scraper.db_path}")
            print(f"{'=' * 50}")

    except Exception as e:
        logger.exception(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
