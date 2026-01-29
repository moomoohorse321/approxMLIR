from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import subprocess as sp
import tempfile
import time
from typing import Optional, Callable

import approx_runtime as ar


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_source_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "src" / filename


def default_cgeist_config(root: Optional[Path] = None) -> ar.CgeistConfig:
    root = root or repo_root()
    return ar.CgeistConfig(
        cgeist_path=os.environ.get("CGEIST_PATH", "cgeist"),
        resource_dir=os.environ.get(
            "CGEIST_RESOURCE_DIR",
            str(root / "llvm-project" / "build" / "lib" / "clang" / "18"),
        ),
        include_dirs=[
            os.environ.get(
                "CGEIST_INCLUDE_DIR",
                str(root / "tools" / "cgeist" / "Test" / "polybench" / "utilities"),
            )
        ],
    )


def default_toolchain() -> ar.ToolchainConfig:
    return ar.ToolchainConfig(
        ml_opt_path=os.environ.get("APPROX_OPT_ML", "approx-opt"),
        cpp_opt_path=os.environ.get("APPROX_OPT_CPP", "polygeist-opt"),
    )


def compile_cpp_path_to_annotated_mlir(
    cpp_path: Path,
    *,
    cgeist_config: ar.CgeistConfig,
    toolchain: ar.ToolchainConfig,
) -> str:
    cpp_source = cpp_path.read_text(encoding="utf-8")
    return ar.compile_cpp_source(
        cpp_source,
        emit="annotated",
        cgeist_config=cgeist_config,
        toolchain=toolchain,
    )


def compile_mlir_to_native_exec(
    mlir_text: str,
    *,
    cgeist_config: ar.CgeistConfig,
    toolchain: ar.ToolchainConfig,
    exec_dir: Optional[Path] = None,
    tag: str = "run",
) -> Path:
    transformed = ar.compile(
        mlir_text,
        workload=ar.WorkloadType.CPP,
        toolchain=toolchain,
    )

    root = exec_dir or (Path(__file__).resolve().parent / "exec")
    run_dir = root / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    mlir_path = run_dir / "module.mlir"
    exec_path = run_dir / "a.out"
    mlir_path.write_text(transformed, encoding="utf-8")

    cmd = [cgeist_config.cgeist_path]
    if cgeist_config.resource_dir:
        cmd.append(f"-resource-dir={cgeist_config.resource_dir}")
    for inc in cgeist_config.include_dirs:
        cmd.extend(["-I", inc])
    cmd.extend(["-lm", str(mlir_path), "-import-mlir", "-o", str(exec_path)])
    result = sp.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cgeist import failed:\n{result.stderr}")

    return exec_path


@dataclass(frozen=True)
class ExecResult:
    elapsed_ms: float
    stdout: str
    stderr: str
    returncode: int


def result_logger(tag: str) -> Callable[[dict, float, float], None]:
    log_dir = Path(__file__).resolve().parent / "exec" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "results.csv"
    header_written = log_path.exists()

    def _callback(cfg: dict, time_ms: float, accuracy: float) -> None:
        nonlocal header_written
        flat_cfg = dict(cfg)
        fieldnames = ["time_ms", "accuracy"] + sorted(flat_cfg.keys())
        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not header_written:
                writer.writeheader()
                header_written = True
            row = {"time_ms": time_ms, "accuracy": accuracy}
            row.update(flat_cfg)
            writer.writerow(row)

    return _callback


def exact_decision_config(params: dict) -> dict:
    exact_config = {}
    for name, param in params.items():
        if getattr(param, "param_type", None) == "decision":
            exact_config[name] = 0
    return exact_config


def run_exec(
    exec_path: Path,
    argv: list[str],
    *,
    stdin: Optional[str] = None,
    work_dir: Optional[Path] = None,
    files: Optional[dict[str, bytes | str]] = None,
) -> ExecResult:
    if work_dir is None:
        work_dir = exec_path.parent
    if files:
        for relpath, content in files.items():
            out_path = work_dir / relpath
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                out_path.write_bytes(content)
            else:
                out_path.write_text(content, encoding="utf-8")

    start = time.perf_counter()
    result = sp.run(
        [str(exec_path), *argv],
        cwd=work_dir,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ExecResult(
        elapsed_ms=elapsed_ms,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )
