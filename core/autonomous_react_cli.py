#!/usr/bin/env python3
"""
Autonomous ReAct CLI - User-friendly interface for the autonomous ReAct agent.

This CLI provides easy access to the autonomous ReAct capabilities with:
1. Intuitive commands and help system
2. Interactive tutorials and examples
3. Error guidance and recovery suggestions
4. Multiple output formats (JSON, markdown, interactive)
5. Configuration management and profiles
"""

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

# Rich library for beautiful CLI output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.progress import SpinnerColumn
    from rich.progress import TextColumn
    from rich.prompt import Confirm
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    # Fallback console
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

        def rule(self, title):
            print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


console = Console()

# Import our autonomous analyzer
from gterminal.core.autonomous_project_analyzer import SimpleAutonomousAnalyzer


class AutonomousReactCLI:
    """User-friendly CLI for the autonomous ReAct agent."""

    def __init__(self):
        self.console = console
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load CLI configuration."""
        config_file = Path.home() / ".autonomous_react_config.json"

        default_config = {
            "default_profile": "business",
            "default_autonomy_level": "semi_auto",
            "output_format": "rich" if HAS_RICH else "plain",
            "project_roots": [],
            "last_analysis_date": None,
        }

        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception:
                pass

        return default_config

    def _save_config(self):
        """Save CLI configuration."""
        config_file = Path.home() / ".autonomous_react_config.json"
        try:
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.console.print(f"⚠️ Warning: Could not save config: {e}")

    async def analyze_command(self, args) -> bool:
        """Handle the analyze command."""
        project_path = Path(args.project or os.getcwd()).resolve()

        if not project_path.exists():
            self.console.print(f"❌ Error: Project path does not exist: {project_path}")
            return False

        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Analyzing project...", total=None)

                analyzer = SimpleAutonomousAnalyzer(project_path)
                report = await analyzer.generate_comprehensive_report()

                progress.update(task, completed=True)
        else:
            self.console.print("🔄 Analyzing project...")
            analyzer = SimpleAutonomousAnalyzer(project_path)
            report = await analyzer.generate_comprehensive_report()

        # Display results based on format
        if args.output == "json":
            print(json.dumps(report, indent=2))
        elif args.output == "markdown":
            self._display_markdown_report(report)
        else:
            self._display_rich_report(report, project_path)

        # Save to file if requested
        if args.save:
            output_file = Path(args.save)
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            self.console.print(f"💾 Report saved to: {output_file}")

        # Update config
        self.config["last_analysis_date"] = datetime.now().isoformat()
        if str(project_path) not in self.config["project_roots"]:
            self.config["project_roots"].append(str(project_path))
        self._save_config()

        return True

    def _display_rich_report(self, report: dict[str, Any], project_path: Path):
        """Display report with rich formatting."""
        if not HAS_RICH:
            self._display_plain_report(report, project_path)
            return

        summary = report["executive_summary"]

        # Title panel
        self.console.print()
        self.console.print(
            Panel(
                f"[bold blue]Autonomous ReAct Analysis Report[/bold blue]\n"
                f"Project: [green]{project_path.name}[/green]\n"
                f"Generated: [dim]{report['generated_at']}[/dim]",
                title="🤖 Analysis Complete",
                border_style="blue",
            )
        )

        # Project scale table
        scale_table = Table(title="📁 Project Scale")
        scale_table.add_column("Metric", style="cyan")
        scale_table.add_column("Count", style="green")

        scale = summary["project_scale"]
        scale_table.add_row("Directories", str(scale["total_directories"]))
        scale_table.add_row("Python Files", str(scale["total_python_files"]))
        scale_table.add_row("Documentation Files", str(scale["total_markdown_files"]))

        self.console.print(scale_table)

        # Consolidation opportunities
        self.console.print()
        consolidation = summary["consolidation_potential"]
        self.console.print(
            Panel(
                f"🔄 [yellow]Consolidation Opportunities[/yellow]\n\n"
                f"• Duplicate Patterns: [red]{consolidation['duplicate_patterns']}[/red]\n"
                f"• Consolidation Targets: [yellow]{consolidation['consolidation_targets']}[/yellow]\n"
                f"• Estimated File Reduction: [green]{consolidation['estimated_file_reduction']}[/green]",
                border_style="yellow",
            )
        )

        # Phase 2 plan
        phase2 = summary["phase2_scope"]
        self.console.print(
            Panel(
                f"🚀 [blue]Phase 2 Implementation[/blue]\n\n"
                f"• Key Features: [green]{phase2['key_features']}[/green]\n"
                f"• Integration Steps: [blue]{phase2['integration_steps']}[/blue]\n"
                f"• Timeline: [magenta]{phase2['estimated_timeline']}[/magenta]",
                border_style="blue",
            )
        )

        # Top recommendations
        self.console.print()
        rec_table = Table(title="🎯 Top Recommendations")
        rec_table.add_column("Priority", style="bold")
        rec_table.add_column("Action", style="cyan")
        rec_table.add_column("Description", style="dim")

        for rec in report["recommendations"][:5]:
            priority_color = {
                "IMMEDIATE": "red",
                "HIGH": "yellow",
                "MEDIUM": "blue",
                "LOW": "green",
            }.get(rec["priority"], "white")

            rec_table.add_row(
                f"[{priority_color}]{rec['priority']}[/{priority_color}]",
                rec["action"],
                rec["description"][:80] + "..."
                if len(rec["description"]) > 80
                else rec["description"],
            )

        self.console.print(rec_table)

        # Next steps
        self.console.print()
        self.console.print("[bold green]✅ Next Steps:[/bold green]")
        for i, step in enumerate(report["next_steps"][:7], 1):
            self.console.print(f"   [dim]{i}.[/dim] {step}")

    def _display_plain_report(self, report: dict[str, Any], project_path: Path):
        """Display report in plain text."""
        summary = report["executive_summary"]

        print("\n" + "=" * 60)
        print("🤖 AUTONOMOUS REACT ANALYSIS REPORT")
        print(f"Project: {project_path.name}")
        print(f"Generated: {report['generated_at']}")
        print("=" * 60)

        # Project scale
        scale = summary["project_scale"]
        print("\n📁 PROJECT SCALE:")
        print(f"   Directories: {scale['total_directories']}")
        print(f"   Python Files: {scale['total_python_files']}")
        print(f"   Documentation Files: {scale['total_markdown_files']}")

        # Consolidation
        consolidation = summary["consolidation_potential"]
        print("\n🔄 CONSOLIDATION OPPORTUNITIES:")
        print(f"   Duplicate Patterns: {consolidation['duplicate_patterns']}")
        print(f"   Consolidation Targets: {consolidation['consolidation_targets']}")
        print(f"   Estimated File Reduction: {consolidation['estimated_file_reduction']}")

        # Phase 2
        phase2 = summary["phase2_scope"]
        print("\n🚀 PHASE 2 IMPLEMENTATION:")
        print(f"   Key Features: {phase2['key_features']}")
        print(f"   Integration Steps: {phase2['integration_steps']}")
        print(f"   Timeline: {phase2['estimated_timeline']}")

        # Recommendations
        print("\n🎯 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"][:5], 1):
            print(f"   {i}. [{rec['priority']}] {rec['action']}")
            print(f"      {rec['description']}")

        # Next steps
        print("\n✅ NEXT STEPS:")
        for i, step in enumerate(report["next_steps"][:7], 1):
            print(f"   {i}. {step}")

    def _display_markdown_report(self, report: dict[str, Any]):
        """Display report in markdown format."""
        summary = report["executive_summary"]

        markdown_content = f"""# Autonomous ReAct Analysis Report

Generated: {report["generated_at"]}

## Executive Summary

### Project Scale
- **Directories**: {summary["project_scale"]["total_directories"]}
- **Python Files**: {summary["project_scale"]["total_python_files"]}
- **Documentation Files**: {summary["project_scale"]["total_markdown_files"]}

### Consolidation Opportunities
- **Duplicate Patterns**: {summary["consolidation_potential"]["duplicate_patterns"]}
- **Consolidation Targets**: {summary["consolidation_potential"]["consolidation_targets"]}
- **Estimated File Reduction**: {summary["consolidation_potential"]["estimated_file_reduction"]}

### Phase 2 Implementation
- **Key Features**: {summary["phase2_scope"]["key_features"]}
- **Integration Steps**: {summary["phase2_scope"]["integration_steps"]}
- **Timeline**: {summary["phase2_scope"]["estimated_timeline"]}

## Top Recommendations

"""

        for i, rec in enumerate(report["recommendations"][:5], 1):
            markdown_content += (
                f"{i}. **[{rec['priority']}]** {rec['action']}\n   {rec['description']}\n\n"
            )

        markdown_content += "## Next Steps\n\n"
        for i, step in enumerate(report["next_steps"][:7], 1):
            markdown_content += f"{i}. {step}\n"

        if HAS_RICH:
            self.console.print(Markdown(markdown_content))
        else:
            print(markdown_content)

    async def demo_command(self, args) -> bool:
        """Handle the demo command."""
        self.console.print()
        self.console.print(
            Panel(
                "[bold green]🎯 Autonomous ReAct Agent Demo[/bold green]\n\n"
                "This demo shows the autonomous ReAct agent analyzing a project structure,\n"
                "identifying consolidation opportunities, and creating implementation plans.",
                title="Demo Mode",
                border_style="green",
            )
        )

        if (
            HAS_RICH
            and not args.no_interactive
            and not Confirm.ask("Run demo analysis on current project?")
        ):
            self.console.print("Demo cancelled.")
            return True

        # Run demo analysis
        project_path = Path(args.project or os.getcwd()).resolve()

        self.console.print(f"\n🔄 Running demo analysis on: [cyan]{project_path.name}[/cyan]")

        try:
            analyzer = SimpleAutonomousAnalyzer(project_path)

            # Show step-by-step process
            self.console.print("\n📋 [yellow]Step 1:[/yellow] Analyzing project structure...")
            await analyzer.analyze_project_structure()

            self.console.print(
                "📋 [yellow]Step 2:[/yellow] Identifying consolidation opportunities..."
            )
            await analyzer.identify_consolidation_opportunities()

            self.console.print("📋 [yellow]Step 3:[/yellow] Creating Phase 2 implementation plan...")
            await analyzer.create_phase2_implementation_plan()

            self.console.print("📋 [yellow]Step 4:[/yellow] Generating comprehensive report...")
            report = await analyzer.generate_comprehensive_report()

            self.console.print("\n✅ [green]Demo completed successfully![/green]")

            # Show key insights
            summary = report["executive_summary"]
            self.console.print("\n🔍 [blue]Key Insights:[/blue]")
            self.console.print(
                f"   • Found {summary['project_scale']['total_python_files']} Python files"
            )
            self.console.print(
                f"   • Identified {summary['consolidation_potential']['duplicate_patterns']} duplication patterns"
            )
            self.console.print(
                f"   • Created {summary['phase2_scope']['integration_steps']}-step implementation plan"
            )

            if HAS_RICH and not args.no_interactive and Confirm.ask("\nShow full analysis report?"):
                self._display_rich_report(report, project_path)

            return True

        except Exception as e:
            self.console.print(f"❌ Demo failed: {e}")
            return False

    def help_command(self, args) -> bool:
        """Handle the help command."""
        if HAS_RICH:
            help_content = """
# Autonomous ReAct Agent CLI

A powerful autonomous agent that uses the ReAct (Reason, Act, Observe) pattern to analyze projects,
identify improvements, and create implementation plans.

## Quick Start

```bash
# Analyze current project
autonomous-react analyze

# Run interactive demo
autonomous-react demo

# Analyze specific project with JSON output
autonomous-react analyze --project /path/to/project --output json

# Get help for specific command
autonomous-react analyze --help
```

## Key Features

- **Autonomous Analysis**: Automatically analyzes project structure and identifies patterns
- **Consolidation Planning**: Finds duplication and suggests consolidation strategies
- **Phase 2 Planning**: Creates implementation plans for enhanced capabilities
- **Multiple Output Formats**: Rich terminal output, JSON, or markdown
- **Interactive Mode**: Step-by-step guidance and confirmations

## Commands

- `analyze` - Analyze project structure and create improvement plans
- `demo` - Run interactive demonstration of capabilities
- `help` - Show this help information
- `config` - Manage CLI configuration

## Examples

```bash
# Basic analysis
autonomous-react analyze

# Save detailed report
autonomous-react analyze --save analysis_report.json

# Markdown output
autonomous-react analyze --output markdown

# Demo mode
autonomous-react demo --no-interactive
```
            """
            self.console.print(Markdown(help_content))
        else:
            print(
                """
AUTONOMOUS REACT AGENT CLI

Commands:
  analyze    Analyze project structure and create improvement plans
  demo       Run interactive demonstration
  help       Show help information
  config     Manage configuration

Quick Start:
  autonomous-react analyze          # Analyze current project
  autonomous-react demo            # Run demonstration
  autonomous-react analyze --help  # Get detailed help

For more information, install the rich library for enhanced output:
  pip install rich
            """
            )

        return True

    def config_command(self, args) -> bool:
        """Handle the config command."""
        if args.list:
            self.console.print("\n📋 [blue]Current Configuration:[/blue]")
            for key, value in self.config.items():
                self.console.print(f"   {key}: [green]{value}[/green]")

        elif args.set:
            try:
                key, value = args.set.split("=", 1)
                # Try to parse as JSON, otherwise treat as string
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string

                self.config[key] = value
                self._save_config()
                self.console.print(f"✅ Set {key} = {value}")
            except ValueError:
                self.console.print("❌ Invalid format. Use: key=value")
                return False

        elif args.reset:
            if HAS_RICH:
                if Confirm.ask("Reset all configuration to defaults?"):
                    self.config = self._load_config()
                    self._save_config()
                    self.console.print("✅ Configuration reset to defaults")
            else:
                print("Configuration would be reset (interactive mode not available)")

        else:
            self.console.print("Use --list, --set key=value, or --reset")

        return True

    async def run(self, args_list: list[str] | None = None) -> bool:
        """Run the CLI with given arguments."""
        parser = self._create_parser()
        args = parser.parse_args(args_list)

        try:
            if args.command == "analyze":
                return await self.analyze_command(args)
            elif args.command == "demo":
                return await self.demo_command(args)
            elif args.command == "help":
                return self.help_command(args)
            elif args.command == "config":
                return self.config_command(args)
            else:
                parser.print_help()
                return True

        except KeyboardInterrupt:
            self.console.print("\n⚠️ Operation cancelled by user")
            return False
        except Exception as e:
            self.console.print(f"❌ Unexpected error: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return False

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="autonomous-react",
            description="Autonomous ReAct Agent CLI - Intelligent project analysis and planning",
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug output")

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Analyze command
        analyze_parser = subparsers.add_parser("analyze", help="Analyze project structure")
        analyze_parser.add_argument(
            "--project", "-p", help="Project path (default: current directory)"
        )
        analyze_parser.add_argument(
            "--output",
            "-o",
            choices=["rich", "json", "markdown"],
            default=self.config.get("output_format", "rich"),
            help="Output format",
        )
        analyze_parser.add_argument("--save", "-s", help="Save detailed report to file")

        # Demo command
        demo_parser = subparsers.add_parser("demo", help="Run interactive demonstration")
        demo_parser.add_argument(
            "--project", "-p", help="Project path (default: current directory)"
        )
        demo_parser.add_argument(
            "--no-interactive", action="store_true", help="Run without prompts"
        )

        # Help command
        subparsers.add_parser("help", help="Show detailed help")

        # Config command
        config_parser = subparsers.add_parser("config", help="Manage configuration")
        config_group = config_parser.add_mutually_exclusive_group()
        config_group.add_argument("--list", action="store_true", help="List current configuration")
        config_group.add_argument("--set", help="Set configuration value (key=value)")
        config_group.add_argument("--reset", action="store_true", help="Reset to defaults")

        return parser


async def main():
    """Main CLI entry point."""
    # Handle no arguments case
    if len(sys.argv) == 1:
        sys.argv.append("help")

    cli = AutonomousReactCLI()
    success = await cli.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
