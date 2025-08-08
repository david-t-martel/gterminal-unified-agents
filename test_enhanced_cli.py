#!/usr/bin/env python3
"""Test script for Enhanced Gemini CLI - Demonstrates all integrated features."""

import asyncio
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    console.print(f"\n[bold yellow]Running:[/bold yellow] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def test_basic_functionality():
    """Test basic CLI functionality."""
    console.print(Panel("[bold]Testing Basic Functionality[/bold]", style="cyan"))
    
    # Test help
    code, stdout, stderr = run_command(["python", "enhanced_gemini_cli.py", "--help"])
    if code == 0:
        console.print("[green]✓[/green] Help command works")
    else:
        console.print(f"[red]✗[/red] Help command failed: {stderr}")
    
    # Test version
    code, stdout, stderr = run_command(["python", "enhanced_gemini_cli.py", "--version"])
    if code == 0 and "2.0.0-enhanced" in stdout:
        console.print("[green]✓[/green] Version shows enhanced CLI")
    else:
        console.print("[red]✗[/red] Version check failed")


def test_profile_management():
    """Test GCP profile management."""
    console.print(Panel("[bold]Testing Profile Management[/bold]", style="cyan"))
    
    # Test profile listing
    code, stdout, stderr = run_command(["python", "enhanced_gemini_cli.py", "profiles"])
    if code == 0:
        console.print("[green]✓[/green] Profile listing works")
        console.print(Syntax(stdout, "text", theme="monokai", line_numbers=False))
    else:
        console.print(f"[red]✗[/red] Profile listing failed: {stderr}")
    
    # Test profile switching
    code, stdout, stderr = run_command(["python", "enhanced_gemini_cli.py", "switch-profile", "business"])
    if code == 0:
        console.print("[green]✓[/green] Profile switching works")
    else:
        console.print(f"[red]✗[/red] Profile switching failed: {stderr}")


def test_super_analysis():
    """Test Super Gemini Agents analysis."""
    console.print(Panel("[bold]Testing Super Gemini Analysis[/bold]", style="cyan"))
    
    # Test on current directory
    code, stdout, stderr = run_command([
        "python", "enhanced_gemini_cli.py", 
        "super-analyze", ".",
        "--type", "architecture",
        "--focus", "performance",
        "--focus", "security"
    ])
    
    if code == 0:
        console.print("[green]✓[/green] Super analysis completed")
        # Show partial output
        lines = stdout.split('\n')
        for line in lines[:20]:  # Show first 20 lines
            console.print(line)
        if len(lines) > 20:
            console.print(f"[dim]... and {len(lines) - 20} more lines[/dim]")
    else:
        console.print(f"[red]✗[/red] Super analysis failed: {stderr}")


def test_orchestration():
    """Test multi-agent orchestration."""
    console.print(Panel("[bold]Testing Multi-Agent Orchestration[/bold]", style="cyan"))
    
    # Test orchestration with a complex task
    code, stdout, stderr = run_command([
        "python", "enhanced_gemini_cli.py",
        "orchestrate",
        "Analyze this project for security vulnerabilities and generate a comprehensive report"
    ])
    
    if code == 0:
        console.print("[green]✓[/green] Orchestration completed")
    else:
        console.print(f"[red]✗[/red] Orchestration failed: {stderr}")


def test_integration_status():
    """Test infrastructure integration."""
    console.print(Panel("[bold]Testing Infrastructure Integration[/bold]", style="cyan"))
    
    code, stdout, stderr = run_command([
        "python", "enhanced_gemini_cli.py",
        "test-integration"
    ])
    
    if code == 0:
        console.print("[green]✓[/green] Integration test completed")
        console.print(Syntax(stdout, "text", theme="monokai", line_numbers=False))
    else:
        console.print(f"[red]✗[/red] Integration test failed: {stderr}")


def test_metrics_dashboard():
    """Test metrics dashboard."""
    console.print(Panel("[bold]Testing Metrics Dashboard[/bold]", style="cyan"))
    
    code, stdout, stderr = run_command([
        "python", "enhanced_gemini_cli.py",
        "metrics"
    ])
    
    if code == 0:
        console.print("[green]✓[/green] Metrics dashboard displayed")
        console.print(Syntax(stdout, "text", theme="monokai", line_numbers=False))
    else:
        console.print(f"[red]✗[/red] Metrics display failed: {stderr}")


async def test_programmatic_usage():
    """Test programmatic usage of enhanced CLI."""
    console.print(Panel("[bold]Testing Programmatic Usage[/bold]", style="cyan"))
    
    try:
        # Import and use directly
        from enhanced_gemini_cli import EnhancedGeminiCLI
        
        # Create instance
        cli = EnhancedGeminiCLI(debug=False)
        
        # Test profile info
        profile_info = cli.get_active_profile()
        console.print(f"[green]✓[/green] Active profile: {profile_info['name']}")
        
        # Test super analysis (if path exists)
        test_path = Path(".")
        if test_path.exists():
            results = await cli.analyze_with_super_agents(
                str(test_path),
                analysis_type="architecture"
            )
            console.print(f"[green]✓[/green] Programmatic analysis completed")
            console.print(f"  Results: {len(results)} analyses")
        
        # Show metrics
        cli.show_metrics_dashboard()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Programmatic usage failed: {e}")


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold cyan]Enhanced Gemini CLI Test Suite[/bold cyan]\n"
        "[dim]Testing all integrated features[/dim]",
        border_style="cyan"
    ))
    
    # Run tests
    test_basic_functionality()
    test_profile_management()
    test_integration_status()
    test_super_analysis()
    test_orchestration()
    test_metrics_dashboard()
    
    # Test programmatic usage
    console.print("\n")
    asyncio.run(test_programmatic_usage())
    
    console.print("\n[bold green]✓ All tests completed![/bold green]")


if __name__ == "__main__":
    main()