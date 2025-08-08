#!/usr/bin/env python3
"""
Use Gemini test writer agent to create tests for auto-claude.
"""

from pathlib import Path
import re

from gemini_config import get_model_for_task


def generate_auto_claude_tests() -> None:
    """Generate comprehensive tests for auto-claude using Gemini."""

    model = get_model_for_task("test_generation")
    target_project = "/home/david/projects/auto-claude"

    print("🧪 Using Gemini to generate auto-claude tests...")

    # Read key files to understand functionality
    files_to_test = [
        ("auto-claude", "test_auto_claude.py"),
        (".auto-claude.yaml", "test_config.py"),
        ("install-auto-claude.sh", "test_installation.py"),
    ]

    tests_dir = Path(target_project) / "tests"
    tests_dir.mkdir(exist_ok=True)

    for source_file, test_file in files_to_test:
        source_path = Path(target_project) / source_file
        if source_path.exists():
            # Read source content
            try:
                content = source_path.read_text()[:3000]  # First 3000 chars
            except:
                content = f"Binary file or script: {source_file}"

            prompt = f"""
            Generate comprehensive pytest tests for this auto-claude component.

            File: {source_file}
            Content preview:
            ```
            {content}
            ```

            Create tests that cover:
            1. Main functionality
            2. Edge cases and error handling
            3. Configuration parsing
            4. Integration scenarios
            5. CLI argument parsing (if applicable)

            Use pytest best practices including:
            - Fixtures for setup/teardown
            - Parametrized tests for multiple scenarios
            - Mocking external dependencies
            - Clear test names and docstrings

            Include imports and all necessary test code.
            """

            response = model.generate_content(prompt)

            # Extract test code from response
            test_content = response.text

            # Clean up the response to get just the code
            if "```python" in test_content:
                code_blocks = re.findall(r"```python\n(.*?)\n```", test_content, re.DOTALL)
                if code_blocks:
                    test_content = code_blocks[0]

            # Save test file
            test_path = tests_dir / test_file
            test_path.write_text(test_content)
            print(f"✅ Created {test_path.name}")

    # Create a test runner script
    runner_content = """#!/usr/bin/env python3
\"\"\"
Test runner for auto-claude tests.
\"\"\"

import sys
import pytest

def main():
    \"\"\"Run all tests.\"\"\"
    # Run pytest with coverage
    exit_code = pytest.main([
        '--verbose',
        '--cov=.',
        '--cov-report=html',
        '--cov-report=term-missing',
        'tests/'
    ])

    return exit_code

if __name__ == '__main__':
    sys.exit(main())
"""

    runner_path = Path(target_project) / "run_tests.py"
    runner_path.write_text(runner_content)
    runner_path.chmod(0o755)
    print(f"✅ Created test runner: {runner_path.name}")

    # Create pytest configuration
    pytest_ini = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
"""

    pytest_ini_path = Path(target_project) / "pytest.ini"
    pytest_ini_path.write_text(pytest_ini)
    print(f"✅ Created pytest configuration: {pytest_ini_path.name}")

    # Create requirements-test.txt
    test_requirements = """pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-asyncio>=0.21.0
"""

    req_path = Path(target_project) / "requirements-test.txt"
    req_path.write_text(test_requirements)
    print(f"✅ Created test requirements: {req_path.name}")

    print("\n✨ Test generation complete!")
    print("\nTo run tests:")
    print(f"  cd {target_project}")
    print("  pip install -r requirements-test.txt")
    print("  python run_tests.py")


if __name__ == "__main__":
    generate_auto_claude_tests()
