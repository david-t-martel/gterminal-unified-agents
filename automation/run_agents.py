#!/usr/bin/env python3
"""Standalone runner for automation agents.
This avoids the app initialization issues.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent directory to path to import automation modules
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))


async def run_test_generation():
    """Run test generation agent."""
    from automation.test_writer_agent import generate_all_missing_tests

    return await generate_all_missing_tests()


async def run_documentation_generation():
    """Run documentation generation agent."""
    from automation.documentation_agent import generate_all_docs

    return await generate_all_docs()


async def run_code_review():
    """Run code review agent."""
    from automation.code_review_agent import review_all_changes

    return await review_all_changes()


async def run_full_automation():
    """Run all automation agents."""
    from automation.fix_orchestrator_agent import run_full_automation

    return await run_full_automation()


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        sys.exit(1)

    command = sys.argv[1]

    if command == "test":
        asyncio.run(run_test_generation())
    elif command == "docs":
        asyncio.run(run_documentation_generation())
    elif command == "review":
        asyncio.run(run_code_review())
    elif command == "full":
        asyncio.run(run_full_automation())
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
