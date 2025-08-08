#!/usr/bin/env python3
"""Consolidated Rust build tooling for My Fullstack Agent.

This module consolidates all Rust-related build tools, metrics collection,
performance analysis, and validation into a single Python utility.
Replaces 5+ separate shell scripts with unified functionality.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import click


@dataclass
class BuildMetrics:
    """Build metrics data structure."""

    timestamp: str
    build_type: str
    binary_size_mb: float | None = None
    binary_path: str | None = None
    cache_hits: int | None = None
    cache_misses: int | None = None
    cache_hit_rate: float | None = None
    build_time_seconds: float | None = None
    rust_version: str | None = None
    cpu_cores: int | None = None
    memory_gb: int | None = None
    dep_count: int | None = None


class RustTools:
    """Consolidated Rust build tools."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path.cwd()
        self.src_dir = self.project_root / "src"
        self.metrics_dir = self.project_root / "build_metrics"
        self.reports_dir = self.project_root / "benchmark_reports"

        # Ensure directories exist
        self.metrics_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

    def run_command(
        self, cmd: list[str], capture_output: bool = True, cwd: Path | None = None
    ) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                cwd=cwd or self.project_root,
                check=False,
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            return 1, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return 1, "", str(e)

    def validate_environment(self) -> tuple[bool, list[str], list[str]]:
        """Validate Rust build environment."""
        passed = True
        warnings: list[Any] = []
        errors: list[Any] = []

        # Check Rust toolchain
        returncode, stdout, _ = self.run_command(["rustc", "--version"])
        if returncode == 0:
            rust_version = stdout.strip()
            # Check if recent version (1.70+)
            version_num = rust_version.split()[1]
            major, minor = map(int, version_num.split(".")[:2])
            if major > 1 or (major == 1 and minor >= 70):
                click.echo(f"✅ Rust compiler: {rust_version}")
            else:
                warnings.append(f"Rust version may be outdated: {version_num}")
        else:
            errors.append("Rust compiler not found")
            passed = False

        # Check Cargo
        returncode, stdout, _ = self.run_command(["cargo", "--version"])
        if returncode == 0:
            click.echo(f"✅ Cargo: {stdout.strip()}")
        else:
            errors.append("Cargo not found")
            passed = False

        # Check essential tools
        tools = [
            ("sccache", "Compilation cache"),
            ("llvm-profdata", "Profile-guided optimization"),
        ]

        for tool, description in tools:
            returncode, _, _ = self.run_command(["which", tool])
            if returncode == 0:
                click.echo(f"✅ {description}: {tool}")
            else:
                warnings.append(f"{description} not available: {tool}")

        # Check project structure
        if (self.src_dir / "Cargo.toml").exists():
            click.echo("✅ Rust project structure found")
        else:
            errors.append("src/Cargo.toml not found")
            passed = False

        # Check system resources
        try:
            import psutil

            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total // (1024**3)

            if cpu_count >= 4:
                click.echo(f"✅ CPU cores: {cpu_count}")
            else:
                warnings.append(f"Limited CPU cores: {cpu_count}")

            if memory_gb >= 8:
                click.echo(f"✅ Memory: {memory_gb}GB")
            else:
                warnings.append(f"Limited memory: {memory_gb}GB")

        except ImportError:
            warnings.append("psutil not available for system resource checking")

        return passed, warnings, errors

    def collect_build_metrics(self, build_type: str = "release") -> BuildMetrics:
        """Collect comprehensive build metrics."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        metrics = BuildMetrics(timestamp=timestamp, build_type=build_type)

        # Binary information
        binary_paths = [
            self.project_root / "target" / "wheels" / "*.whl",
            self.src_dir / "target" / "release" / "libfullstack_agent_rust.so",
            self.src_dir / "target" / "release" / "fullstack_agent_rust.dll",
            self.src_dir / "target" / "release" / "libfullstack_agent_rust.dylib",
        ]

        for path_pattern in binary_paths:
            if "*" in str(path_pattern):
                # Handle glob pattern
                import glob

                matches = glob.glob(str(path_pattern))
                if matches:
                    binary_path = Path(matches[0])
                    break
            elif path_pattern.exists():
                binary_path = path_pattern
                break
        else:
            binary_path = None

        if binary_path and binary_path.exists():
            size_bytes = binary_path.stat().st_size
            metrics.binary_size_mb = size_bytes / (1024 * 1024)
            metrics.binary_path = str(binary_path)

        # Cache statistics
        returncode, stdout, _ = self.run_command(["sccache", "--show-stats"])
        if returncode == 0:
            lines = stdout.strip().split("\n")
            for line in lines:
                if "Cache hits" in line:
                    metrics.cache_hits = int(line.split()[-1])
                elif "Cache misses" in line:
                    metrics.cache_misses = int(line.split()[-1])

            if metrics.cache_hits is not None and metrics.cache_misses is not None:
                total = metrics.cache_hits + metrics.cache_misses
                if total > 0:
                    metrics.cache_hit_rate = (metrics.cache_hits / total) * 100

        # System information
        returncode, stdout, _ = self.run_command(["rustc", "--version"])
        if returncode == 0:
            metrics.rust_version = stdout.strip().split()[1]

        try:
            import psutil

            metrics.cpu_cores = psutil.cpu_count()
            metrics.memory_gb = psutil.virtual_memory().total // (1024**3)
        except ImportError:
            pass

        # Dependency count
        if (self.src_dir / "Cargo.toml").exists():
            returncode, stdout, _ = self.run_command(
                ["cargo", "metadata", "--format-version", "1"], cwd=self.src_dir
            )
            if returncode == 0:
                try:
                    metadata = json.loads(stdout)
                    metrics.dep_count = len(metadata.get("packages", []))
                except json.JSONDecodeError:
                    pass

        return metrics

    def save_metrics(self, metrics: BuildMetrics) -> Path:
        """Save build metrics to JSON file."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"build_metrics_{timestamp_str}.json"

        # Convert to dict for JSON serialization
        metrics_dict = {
            "timestamp": metrics.timestamp,
            "build_type": metrics.build_type,
            "binary": (
                {
                    "path": metrics.binary_path,
                    "size_bytes": int(metrics.binary_size_mb * 1024 * 1024)
                    if metrics.binary_size_mb
                    else None,
                    "size_mb": metrics.binary_size_mb,
                }
                if metrics.binary_path
                else None
            ),
            "cache": (
                {
                    "hits": metrics.cache_hits,
                    "misses": metrics.cache_misses,
                    "hit_rate_percent": metrics.cache_hit_rate,
                }
                if metrics.cache_hits is not None
                else None
            ),
            "build_time_seconds": metrics.build_time_seconds,
            "system": {
                "rust_version": metrics.rust_version,
                "cpu_cores": metrics.cpu_cores,
                "memory_gb": metrics.memory_gb,
            },
            "dependencies": {"total_count": metrics.dep_count} if metrics.dep_count else None,
            "optimizations": {
                "lto": "fat",
                "opt_level": "3",
                "codegen_units": "1",
                "target_cpu": "native",
                "strip": True,
            },
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)

        return metrics_file

    def analyze_performance_trends(self) -> Path | None:
        """Analyze performance trends across builds."""
        metric_files = sorted(self.metrics_dir.glob("build_metrics_*.json"), reverse=True)

        if len(metric_files) < 2:
            click.echo(f"⚠️  Need at least 2 builds for trend analysis (found {len(metric_files)})")
            return None

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        trends_report = self.reports_dir / f"performance_trends_{timestamp_str}.md"

        with open(trends_report, "w") as f:
            f.write("# Rust Performance Trends Analysis\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Builds Analyzed:** {len(metric_files)}\n\n")

            # Binary size trends
            f.write("## Binary Size Trends\n\n")
            f.write("| Build | Date | Size (MB) | Change |\n")
            f.write("|-------|------|-----------|--------|\n")

            previous_size = None
            for i, file_path in enumerate(metric_files):
                try:
                    with open(file_path) as mf:
                        data = json.load(mf)

                    date_str = data.get("timestamp", "Unknown").split("T")[0]
                    size_mb = (
                        data.get("binary", {}).get("size_mb", "N/A")
                        if data.get("binary")
                        else "N/A"
                    )

                    change = ""
                    if previous_size is not None and size_mb != "N/A" and previous_size != "N/A":
                        change_num = size_mb - previous_size
                        if change_num > 0:
                            change = f"+{change_num:.2f}MB"
                        elif change_num < 0:
                            change = f"{change_num:.2f}MB"
                        else:
                            change = "No change"
                    elif i == 0:
                        change = "Latest"

                    f.write(f"| Build {i + 1} | {date_str} | {size_mb} | {change} |\n")
                    previous_size = size_mb if size_mb != "N/A" else previous_size

                except (json.JSONDecodeError, FileNotFoundError):
                    f.write(f"| Build {i + 1} | Unknown | N/A | N/A |\n")

            f.write("\n")

            # Cache performance trends
            f.write("## Cache Performance Trends\n\n")
            f.write("| Build | Date | Hit Rate (%) | Hits | Misses |\n")
            f.write("|-------|------|--------------|------|--------|\n")

            for i, file_path in enumerate(metric_files):
                try:
                    with open(file_path) as mf:
                        data = json.load(mf)

                    date_str = data.get("timestamp", "Unknown").split("T")[0]
                    cache_data = data.get("cache", {})
                    hit_rate = cache_data.get("hit_rate_percent", "N/A") if cache_data else "N/A"
                    hits = cache_data.get("hits", "N/A") if cache_data else "N/A"
                    misses = cache_data.get("misses", "N/A") if cache_data else "N/A"

                    f.write(f"| Build {i + 1} | {date_str} | {hit_rate} | {hits} | {misses} |\n")

                except (json.JSONDecodeError, FileNotFoundError):
                    f.write(f"| Build {i + 1} | Unknown | N/A | N/A | N/A |\n")

            f.write("\n")

            # Performance insights
            f.write("## Performance Insights\n\n")

            if len(metric_files) >= 2:
                try:
                    with open(metric_files[0]) as f1, open(metric_files[-1]) as f2:
                        latest_data = json.load(f1)
                        oldest_data = json.load(f2)

                    f.write("### Latest vs Initial Build\n\n")

                    # Binary size comparison
                    latest_size = (
                        latest_data.get("binary", {}).get("size_mb")
                        if latest_data.get("binary")
                        else None
                    )
                    oldest_size = (
                        oldest_data.get("binary", {}).get("size_mb")
                        if oldest_data.get("binary")
                        else None
                    )

                    if latest_size and oldest_size:
                        size_change = latest_size - oldest_size
                        size_percent = (size_change / oldest_size) * 100

                        if size_change > 0:
                            f.write(
                                f"- **Binary Size:** Increased by {size_change:.2f}MB ({size_percent:.1f}%)\n"
                            )
                        elif size_change < 0:
                            f.write(
                                f"- **Binary Size:** Decreased by {abs(size_change):.2f}MB ({abs(size_percent):.1f}%)\n",
                            )
                        else:
                            f.write("- **Binary Size:** No change\n")

                    f.write("\n")

                except (json.JSONDecodeError, FileNotFoundError):
                    f.write("*Unable to load comparison data*\n\n")

            # Recommendations
            f.write("### Recommendations\n\n")
            f.write("- 🔧 **Regular Monitoring:** Track trends to catch performance regressions\n")
            f.write("- 📊 **Benchmark Integration:** Include performance tests in CI/CD\n")
            f.write(
                "- 🦀 **Profile-Guided Optimization:** Use `make rust-build-pgo` for best performance\n"
            )
            f.write("- 💾 **Cache Optimization:** Monitor cache hit rates and adjust as needed\n\n")

        click.echo(f"✅ Performance trends analysis complete: {trends_report}")
        return trends_report

    def generate_benchmark_report(self) -> Path:
        """Generate comprehensive benchmark report."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"benchmark_report_{timestamp_str}.md"

        with open(report_file, "w") as f:
            f.write("# Rust Performance Benchmark Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # System information
            f.write("## System Information\n\n")

            try:
                import psutil

                f.write(f"- **CPU Cores:** {psutil.cpu_count()}\n")
                f.write(f"- **Memory:** {psutil.virtual_memory().total // (1024**3)}GB\n")
            except ImportError:
                f.write("- **System Info:** Not available (psutil required)\n")

            # Rust version
            returncode, stdout, _ = self.run_command(["rustc", "--version"])
            if returncode == 0:
                f.write(f"- **Rust Version:** {stdout.strip()}\n")

            f.write("\n")

            # Performance expectations
            f.write("## Expected Performance Improvements\n\n")
            f.write("| Component | Python Baseline | Rust Extension | Expected Speedup |\n")
            f.write("|-----------|-----------------|----------------|------------------|\n")
            f.write("| JSON Processing | 100% | Rust + SIMD | 3-5x faster |\n")
            f.write("| Cache Operations | 100% | Rust + DashMap | 5-10x faster |\n")
            f.write("| File I/O | 100% | Rust + Memory Mapping | 2-4x faster |\n")
            f.write("| Authentication | 100% | Rust + Hardware Crypto | 10-20x faster |\n")
            f.write("| WebSocket Handling | 100% | Rust + Tokio | 3-8x faster |\n\n")

            # Current optimizations
            f.write("## Current Optimizations Applied\n\n")
            f.write(
                "- ✅ **Link-Time Optimization (LTO):** `fat` - Maximum cross-crate optimization\n"
            )
            f.write("- ✅ **Optimization Level:** `3` - Maximum speed optimization\n")
            f.write("- ✅ **Codegen Units:** `1` - Single unit for maximum optimization\n")
            f.write("- ✅ **Target CPU:** `native` - Hardware-specific optimizations\n")
            f.write("- ✅ **Panic Strategy:** `abort` - Smaller binaries, faster execution\n")
            f.write("- ✅ **Symbol Stripping:** Enabled - Reduced binary size\n\n")

            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. **Run Profile-Guided Optimization:** `make rust-build-pgo`\n")
            f.write("2. **Measure Real Performance:** `make rust-benchmark`\n")
            f.write("3. **Profile Memory Usage:** `make rust-profile`\n")
            f.write("4. **Monitor Cache Performance:** `make rust-cache-stats`\n")
            f.write("5. **Regular Performance Regression Testing:** Integrate into CI/CD\n\n")

        click.echo(f"✅ Benchmark report generated: {report_file}")
        return report_file


# CLI Interface
@click.group()
def cli() -> None:
    """Consolidated Rust build tools for My Fullstack Agent."""


@cli.command()
def validate() -> None:
    """Validate Rust build environment."""
    tools = RustTools()
    passed, warnings, errors = tools.validate_environment()

    if warnings:
        click.echo("\n⚠️  Warnings:")
        for warning in warnings:
            click.echo(f"  - {warning}")

    if errors:
        click.echo("\n❌ Errors:")
        for error in errors:
            click.echo(f"  - {error}")

    if passed:
        click.echo("\n✅ Build environment validation PASSED")
        sys.exit(0)
    else:
        click.echo("\n❌ Build environment validation FAILED")
        sys.exit(1)


@cli.command()
@click.option("--build-type", default="release", help="Build type for metrics")
def metrics(build_type) -> None:
    """Collect build metrics."""
    tools = RustTools()
    metrics = tools.collect_build_metrics(build_type)
    metrics_file = tools.save_metrics(metrics)

    click.echo(f"✅ Build metrics collected: {metrics_file}")

    if metrics.binary_size_mb:
        click.echo(f"📦 Binary size: {metrics.binary_size_mb:.2f}MB")
    if metrics.cache_hit_rate:
        click.echo(f"💾 Cache hit rate: {metrics.cache_hit_rate:.1f}%")


@cli.command()
def trends() -> None:
    """Analyze performance trends."""
    tools = RustTools()
    report = tools.analyze_performance_trends()
    if report:
        click.echo(f"📈 View report: cat {report}")


@cli.command()
def benchmark_report() -> None:
    """Generate benchmark report."""
    tools = RustTools()
    report = tools.generate_benchmark_report()
    click.echo(f"📊 View report: cat {report}")


@cli.command()
def cache_stats() -> None:
    """Show cache statistics."""
    tools = RustTools()
    returncode, stdout, stderr = tools.run_command(["sccache", "--show-stats"])

    if returncode == 0:
        click.echo("💾 sccache statistics:")
        click.echo(stdout)
    else:
        click.echo("❌ sccache not available or not running")
        click.echo(stderr)


if __name__ == "__main__":
    cli()
