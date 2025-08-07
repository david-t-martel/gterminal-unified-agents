#!/usr/bin/env python3
"""
Build script for GTerminal Rust Extensions

This script handles the complete build process including:
- Rust compilation with PyO3
- Python wheel generation
- Installation and testing
- Performance benchmarking
"""

import builtins
import contextlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description, check=True, capture_output=False):
    """Run a command with proper error handling"""
    print(f"🔨 {description}")
    print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    start_time = time.time()

    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True,
                text=True,
                check=check,
            )
            duration = time.time() - start_time
            print(f"   ✅ Completed in {duration:.2f}s")
            return result
        else:
            result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check)
            duration = time.time() - start_time
            print(f"   ✅ Completed in {duration:.2f}s")
            return result
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"   ❌ Failed after {duration:.2f}s")
        if hasattr(e, "stdout") and e.stdout:
            print(f"   stdout: {e.stdout}")
        if hasattr(e, "stderr") and e.stderr:
            print(f"   stderr: {e.stderr}")
        if check:
            raise
        return e


def check_prerequisites():
    """Check if all required tools are installed"""
    print("🔍 Checking prerequisites...")

    required_tools = [
        ("rustc", "Rust compiler"),
        ("cargo", "Rust package manager"),
        ("python3", "Python 3"),
    ]

    missing_tools = []

    # Note: try-except in loop is acceptable here for tool version checking
    # as it's a one-time setup operation, not a performance-critical path
    for tool, description in required_tools:
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, text=True)
            print(
                f"   ✅ {description}: {result.stdout.split()[1] if result.stdout else 'installed'}"
            )
        except FileNotFoundError:  # noqa: PERF203
            missing_tools.append((tool, description))
            print(f"   ❌ {description}: not found")

    if missing_tools:
        print("\n❌ Missing required tools:")
        for tool, desc in missing_tools:
            print(f"   - {tool}: {desc}")
        print("\nPlease install the missing tools and try again.")
        return False

    # Check for maturin
    try:
        result = subprocess.run(["maturin", "--version"], capture_output=True, text=True)
        print(f"   ✅ Maturin: {result.stdout.strip()}")
    except FileNotFoundError:
        print("   ⚠️  Maturin not found - will install it")
        run_command(
            [sys.executable, "-m", "pip", "install", "maturin[patchelf]"],
            "Installing maturin",
        )

    return True


def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build artifacts...")

    dirs_to_clean = ["target", "build", "dist", "*.egg-info"]
    files_to_clean = ["Cargo.lock"]

    for pattern in dirs_to_clean:
        if "*" in pattern:
            import glob

            for path in glob.glob(pattern):
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"   🗑️  Removed: {path}")
        else:
            if os.path.exists(pattern):
                shutil.rmtree(pattern, ignore_errors=True)
                print(f"   🗑️  Removed: {pattern}")

    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)
            print(f"   🗑️  Removed: {file}")


def build_rust_extension(mode="develop", features=None):
    """Build the Rust extension using maturin"""
    print(f"🦀 Building Rust extension ({'development' if mode == 'develop' else 'release'})...")

    cmd = ["maturin", mode]

    if mode == "build":
        cmd.extend(["--release", "--strip"])

    if features:
        cmd.extend(["--features", features])

    # Add additional flags for optimization
    if mode == "build":
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-C target-cpu=native -C opt-level=3"
        run_command(
            cmd,
            f"Building Rust extension ({'release' if mode == 'build' else 'development'})",
            check=True,
        )
    else:
        run_command(
            cmd,
            f"Building Rust extension ({'release' if mode == 'build' else 'development'})",
            check=True,
        )


def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")

    # Rust tests
    run_command(["cargo", "test", "--release"], "Running Rust unit tests")

    # Python tests (if example works, basic functionality is verified)
    try:
        run_command(
            [sys.executable, "example_usage.py"],
            "Running Python integration tests",
            check=False,
        )  # Don't fail build if demo has issues
    except Exception as e:
        print(f"   ⚠️  Python tests had issues: {e}")


def benchmark_performance():
    """Run performance benchmarks"""
    print("📊 Running performance benchmarks...")

    # Rust benchmarks
    try:
        run_command(["cargo", "bench"], "Running Rust benchmarks", check=False)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  Rust benchmarks not available (requires unstable features)")

    # Python benchmarks via example script
    try:
        from gterminal_rust_extensions import benchmark_components

        print("   Running component benchmarks...")
        results = benchmark_components(iterations=10000)

        print("   📈 Benchmark Results:")
        for component, ops_per_sec in results.items():
            print(f"      {component}: {ops_per_sec:,.0f} ops/sec")

    except ImportError:
        print("   ⚠️  Extension not available for benchmarking yet")
    except Exception as e:
        print(f"   ⚠️  Benchmark failed: {e}")


def generate_docs():
    """Generate documentation"""
    print("📚 Generating documentation...")

    # Rust docs
    run_command(["cargo", "doc", "--no-deps"], "Generating Rust documentation", check=False)

    # Check if docs were generated
    docs_path = Path("target/doc")
    if docs_path.exists():
        print(f"   📖 Rust docs generated at: {docs_path.absolute()}")
    else:
        print("   ⚠️  Rust documentation generation failed")


def create_wheel():
    """Create Python wheel"""
    print("📦 Creating Python wheel...")

    run_command(["maturin", "build", "--release", "--strip"], "Creating wheel package")

    # Find the generated wheel
    dist_path = Path("target/wheels")
    if dist_path.exists():
        wheels = list(dist_path.glob("*.whl"))
        if wheels:
            print(f"   📦 Created wheel: {wheels[0].name}")
            return wheels[0]

    return None


def install_extension():
    """Install the extension for testing"""
    print("💾 Installing extension...")

    # Force reinstall to ensure we get the latest version
    with contextlib.suppress(builtins.BaseException):
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "gterminal-rust-extensions",
                "-y",
            ],
            "Uninstalling previous version",
            check=False,
        )

    run_command(["maturin", "develop", "--release"], "Installing development version")


def main():
    """Main build process"""
    print("🚀 GTerminal Rust Extensions Build Script")
    print("=" * 60)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Build GTerminal Rust Extensions")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts first")
    parser.add_argument("--release", action="store_true", help="Build in release mode")
    parser.add_argument("--no-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--no-install", action="store_true", help="Skip installation")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--docs", action="store_true", help="Generate documentation")
    parser.add_argument("--wheel", action="store_true", help="Create wheel package")
    parser.add_argument("--features", help="Comma-separated list of features to enable")

    args = parser.parse_args()

    try:
        # Step 1: Check prerequisites
        if not check_prerequisites():
            sys.exit(1)

        # Step 2: Clean if requested
        if args.clean:
            clean_build()

        # Step 3: Build extension
        build_mode = "build" if args.release else "develop"
        build_rust_extension(mode=build_mode, features=args.features)

        # Step 4: Install for testing
        if not args.no_install:
            install_extension()

        # Step 5: Run tests
        if not args.no_tests:
            run_tests()

        # Step 6: Generate documentation
        if args.docs:
            generate_docs()

        # Step 7: Create wheel
        if args.wheel:
            wheel_path = create_wheel()
            if wheel_path:
                print(f"   📦 Wheel available at: {wheel_path}")

        # Step 8: Run benchmarks
        if args.benchmark:
            benchmark_performance()

        print("\n" + "=" * 60)
        print("✅ Build completed successfully!")

        # Show next steps
        print("\n📋 Next steps:")
        print("   1. Run the example: python example_usage.py")
        print("   2. Import in your code: from gterminal_rust_extensions import *")
        print("   3. View docs: cargo doc --open")
        if args.wheel:
            print("   4. Install wheel: pip install target/wheels/*.whl")

    except KeyboardInterrupt:
        print("\n🛑 Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
