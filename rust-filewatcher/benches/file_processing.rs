/*!
 * Performance benchmarks for gterminal-filewatcher
 */

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gterminal_filewatcher::{Config, WatchEngine};
use std::path::PathBuf;
use tempfile::TempDir;

fn benchmark_config_creation(c: &mut Criterion) {
    c.bench_function("config_default", |b| {
        b.iter(|| {
            black_box(Config::default())
        })
    });
}

fn benchmark_project_detection(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();

    // Create test project files
    std::fs::write(project_path.join("pyproject.toml"), "[project]\nname = \"test\"").unwrap();
    std::fs::write(project_path.join("package.json"), "{}").unwrap();
    std::fs::write(project_path.join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();

    c.bench_function("project_detection", |b| {
        b.iter(|| {
            black_box(Config::load_from_project(project_path).unwrap())
        })
    });
}

fn benchmark_file_filtering(c: &mut Criterion) {
    let config = Config::default();

    let test_paths = vec![
        PathBuf::from("src/main.py"),
        PathBuf::from("src/utils.ts"),
        PathBuf::from("tests/test_main.py"),
        PathBuf::from("node_modules/package/index.js"),
        PathBuf::from("target/debug/main"),
        PathBuf::from(".git/config"),
        PathBuf::from("dist/bundle.js"),
    ];

    c.bench_function("file_filtering", |b| {
        b.iter(|| {
            for path in &test_paths {
                black_box(config.should_ignore_path(path));
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    black_box(config.should_watch_extension(ext));
                }
            }
        })
    });
}

fn benchmark_watch_engine_creation(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::default();

    c.bench_function("watch_engine_creation", |b| {
        b.iter(|| {
            black_box(WatchEngine::new(
                temp_dir.path().to_path_buf(),
                config.clone(),
                true,
            ).unwrap())
        })
    });
}

fn benchmark_large_directory_scan(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();

    // Create a realistic project structure
    let dirs = ["src", "tests", "docs", "examples"];
    let files_per_dir = 10;

    for dir in &dirs {
        let dir_path = project_path.join(dir);
        std::fs::create_dir_all(&dir_path).unwrap();

        for i in 0..files_per_dir {
            std::fs::write(
                dir_path.join(format!("file_{}.py", i)),
                format!("# Test file {}\ndef function_{}():\n    pass\n", i, i),
            ).unwrap();
        }
    }

    c.bench_function("large_directory_scan", |b| {
        b.iter(|| {
            let config = Config::default();
            // Simulate directory scanning logic
            for entry in walkdir::WalkDir::new(project_path) {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() {
                        black_box(config.should_ignore_path(path));
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            black_box(config.should_watch_extension(ext));
                        }
                    }
                }
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_config_creation,
    benchmark_project_detection,
    benchmark_file_filtering,
    benchmark_watch_engine_creation,
    benchmark_large_directory_scan
);

criterion_main!(benches);
