#!/usr/bin/env python3
"""
agent.py - simple autoresearch controller (LLM-driven, git-backed).

Requirements:
  pip install GitPython openai

Features:
  - reads program.md (instructions) and train.py (current code)
  - asks LLM to return a full improved train.py (only the file content)
  - validates candidate (syntax, forbidden patterns)
  - runs candidate for a short time (timeout)
  - parses validation metric (val_bpb) from stdout
  - if improved -> commit; else revert
  - logs results to results.csv and agent.log
"""

import argparse
import ast
import csv
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Optional, Tuple

import git  # GitPython
import openai

# Configure logger
logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("agent")

# ---------------------------
# Configuration / constants
# ---------------------------
PROMPT_FILE = "program.md"
TARGET_FILE = "train.py"
CANDIDATE_FILE = "train.candidate.py"
RESULTS_CSV = "results.csv"
BACKUP_REF = "refs/heads/agent-backup"  # not strictly required but helpful
DEFAULT_RUN_TIMEOUT = 300  # seconds per training invocation
DEFAULT_MAX_ITERS = 1000
LLM_RETRY_MAX = 4
LLM_RETRY_BASE = 1.0

# Simple forbidden patterns to prevent LLM from touching dataset or removing files
FORBIDDEN_PATTERNS = [
    r"open\s*\(\s*['\"]prepare\.py['\"]",   # read/modify prepare.py
    r"open\s*\(\s*['\"]data/",              # any data path access attempt
    r"shutil\.rmtree",
    r"os\.remove",
    r"os\.rmdir",
    r"requests\.",
    r"urllib\.",
    r"subprocess\.run\(",
    r"exec\(",
    r"eval\(",
    # add more if you have custom files to protect
]

# Regex to extract val_bpb from stdout lines such as:
# "val_bpb: 1.234567" or "val bpb = 1.2345" or "validation bits-per-byte: 1.234"
BPB_PATTERNS = [
    re.compile(r"val[_\s-]*bpb[:=]\s*([0-9]+\.[0-9]+)"),
    re.compile(r"validation[_\s-]*(?:bits[-\s]*per[-\s]*byte|bpb)[:=]\s*([0-9]+\.[0-9]+)"),
    re.compile(r"val[:\s]*bpb[:\s]*([0-9]+\.[0-9]+)"),
]

# ---------------------------
# Helpers
# ---------------------------

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def extract_code_from_response(text: str) -> str:
    """
    Extract Python code from LLM response prioritizing fenced blocks.
    If none found, return full text.
    """
    # prefer ```python blocks
    match = re.search(r"```(?:python)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Otherwise try triple backticks without language
    match = re.search(r"```([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    # fallback: return everything
    return text.strip()

def candidate_is_safe(code: str) -> Tuple[bool, Optional[str]]:
    """Simple checks: forbidden patterns + valid python syntax."""
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return False, f"forbidden pattern matched: {pat}"
    # syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    return True, None

def parse_bpb(stdout: str) -> Optional[float]:
    """Parse BPB from stdout using common patterns."""
    for line in stdout.splitlines():
        for pat in BPB_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    continue
    # if none found, try to find any floating number on a line containing "val" and "bpb"
    for line in stdout.splitlines():
        if "val" in line.lower() and "bpb" in line.lower():
            m = re.search(r"([0-9]+\.[0-9]+)", line)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    return None

def append_result_row(row: dict) -> None:
    header = ["timestamp", "iteration", "status", "bpb", "commit", "message", "duration_s"]
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------
# LLM interface (OpenAI example)
# ---------------------------

def call_llm_openai(prompt_system: str, prompt_user: str, model: str, max_tokens: int = 2000) -> str:
    """
    Minimal OpenAI chat call. Exponential backoff included.
    Assumes OPENAI_API_KEY is set in env and openai package is installed.
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY") or openai.api_key
    attempt = 0
    backoff = LLM_RETRY_BASE
    while attempt < LLM_RETRY_MAX:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            # OpenAI returns 'choices' -> 0 -> 'message' -> 'content'
            content = resp["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            attempt += 1
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, LLM_RETRY_MAX, str(e))
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("LLM call failed after retries")

def call_llm(prompt_system: str, prompt_user: str, backend: str, model: str) -> str:
    """
    Adapter: currently supports backend == 'openai'. Extend for 'anthropic' or 'ollama'.
    """
    backend = backend.lower()
    if backend == "openai":
        return call_llm_openai(prompt_system, prompt_user, model)
    else:
        raise NotImplementedError(f"LLM backend '{backend}' not implemented. Add adapter.")

# ---------------------------
# Training execution
# ---------------------------

def run_candidate_run(candidate_path: str, timeout: int) -> Tuple[bool, str, Optional[str]]:
    """
    Replace TARGET_FILE with candidate and run training, returning success flag, stdout, and error message (if any).
    Uses git to snapshot and revert in case of crash.
    """
    repo = git.Repo(".")
    # Save current HEAD content of TARGET_FILE (or use git checkout later)
    original_content = None
    try:
        original_content = read_file(TARGET_FILE)
    except FileNotFoundError:
        original_content = None

    # write candidate into TARGET_FILE
    write_file(TARGET_FILE, read_file(candidate_path))

    # Run the training script.
    # It is important to run in a clean env; we pass PYTHONUNBUFFERED so output flushes.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.time()
    try:
        p = subprocess.run(
            [sys.executable, TARGET_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
            text=True,
        )
        duration = time.time() - start
        stdout = p.stdout or ""
        return True, stdout, None
    except subprocess.TimeoutExpired as e:
        # restore original file
        if original_content is not None:
            write_file(TARGET_FILE, original_content)
        else:
            # if no original, leave candidate but log
            logger.warning("Original train.py not found; candidate left in place after timeout.")
        return False, getattr(e, "output", "") or "", f"timeout after {timeout}s"
    except Exception as e:
        # restore original
        if original_content is not None:
            write_file(TARGET_FILE, original_content)
        return False, "", f"runtime error: {e}"

# ---------------------------
# Main loop
# ---------------------------

def main_loop(args):
    repo = git.Repo(".")
    if repo.bare:
        raise RuntimeError("Git repository required in current directory.")

    program_text = read_file(PROMPT_FILE) if os.path.exists(PROMPT_FILE) else ""
    logging.info("Loaded program.md (%d chars)", len(program_text))

    # initialize best_bpb by running baseline if possible
    best_bpb = float("inf")
    initialized = False

    # Try a baseline run (if user wants to skip baseline, they can set --skip-baseline)
    if not args.skip_baseline:
        logger.info("Running baseline training to establish best_bpb...")
        try:
            success, stdout, err = run_candidate_run(TARGET_FILE, timeout=args.run_timeout)
            if success:
                bpb = parse_bpb(stdout)
                if bpb is not None:
                    best_bpb = bpb
                    initialized = True
                    logger.info("Baseline val_bpb=%.6f", bpb)
                    append_result_row({
                        "timestamp": datetime.utcnow().isoformat(),
                        "iteration": 0,
                        "status": "baseline",
                        "bpb": bpb,
                        "commit": "",
                        "message": "baseline run",
                        "duration_s": args.run_timeout,
                    })
                else:
                    logger.warning("Baseline run produced no BPB; continuing with inf baseline.")
            else:
                logger.warning("Baseline run failed: %s", err)
        except Exception as e:
            logger.exception("Baseline failed: %s", e)

    # main iterations
    for iteration in range(1, args.max_iterations + 1):
        iter_start = time.time()
        logger.info("Iteration %d starting (best_bpb=%.6f)", iteration, best_bpb)

        # read current files
        code = read_file(TARGET_FILE)
        prompt_system = (
            "You are an expert ML engineer. You will output a complete, self-contained `train.py` "
            "Python file only (no commentary). The goal is to improve validation BPB (lower is better) "
            "according to the instructions below. Do not modify other files; do not touch datasets. "
            "If modifications would require new files, instead propose code that uses the existing repo. "
            "Return only the code (a Python file) inside a markdown code block or raw."
        )
        prompt_user = (
            f"Research goal (program.md):\n{program_text}\n\n"
            f"Current train.py (for reference):\n{code}\n\n"
            "Constraints:\n"
            "- Only return the full contents of train.py. Wrap in ```python ... ``` or return raw code.\n"
            "- Do not access or modify files outside the training code. Avoid network calls.\n"
            "- Keep runtime changes small (e.g., hyperparameters, small module adjustments); huge rewrites are allowed but may break tests.\n"
            "- If you cannot improve, you may return the same file.\n\n"
            "Return the new version of train.py."
        )

        # Get suggestion from LLM
        try:
            raw_response = call_llm(prompt_system, prompt_user, backend=args.backend, model=args.model)
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            append_result_row({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": iteration,
                "status": "llm_failed",
                "bpb": "",
                "commit": "",
                "message": str(e),
                "duration_s": 0,
            })
            # small sleep and continue
            time.sleep(5)
            continue

        candidate_code = extract_code_from_response(raw_response)

        # write candidate file
        write_file(CANDIDATE_FILE, candidate_code)

        # safety checks
        safe, reason = candidate_is_safe(candidate_code)
        if not safe:
            logger.warning("Candidate rejected: %s", reason)
            append_result_row({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": iteration,
                "status": "rejected_safety",
                "bpb": "",
                "commit": "",
                "message": reason,
                "duration_s": time.time() - iter_start,
            })
            continue

        # run candidate (overwrites TARGET_FILE internally)
        success, stdout, err = run_candidate_run(CANDIDATE_FILE, timeout=args.run_timeout)
        run_duration = time.time() - iter_start
        if not success:
            logger.warning("Candidate run failed or timed out: %s", err)
            append_result_row({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": iteration,
                "status": "run_failed",
                "bpb": "",
                "commit": "",
                "message": err or "run_failed",
                "duration_s": run_duration,
            })
            # ensure git reset to HEAD to avoid partial changes
            try:
                repo.git.reset("--hard", "HEAD")
            except Exception:
                pass
            continue

        # parse metric
        bpb = parse_bpb(stdout)
        if bpb is None:
            logger.warning("Could not parse BPB from run stdout.")
            append_result_row({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": iteration,
                "status": "no_bpb",
                "bpb": "",
                "commit": "",
                "message": "no_bpb_parsed",
                "duration_s": run_duration,
            })
            # revert to last commit since we cannot evaluate
            try:
                repo.git.reset("--hard", "HEAD")
            except Exception:
                pass
            continue

        # Evaluate improvement (smaller is better)
        improved = bpb < best_bpb - args.min_delta
        if improved:
            # commit change
            try:
                repo.git.add(TARGET_FILE)
                commit_msg = f"autoresearch: val_bpb={bpb:.6f} iter={iteration}"
                repo.index.commit(commit_msg)
                commit_hash = repo.head.commit.hexsha
                best_bpb = bpb
                logger.info("Improved: val_bpb=%.6f -> committed %s", bpb, commit_hash)
                append_result_row({
                    "timestamp": datetime.utcnow().isoformat(),
                    "iteration": iteration,
                    "status": "improved",
                    "bpb": bpb,
                    "commit": commit_hash,
                    "message": commit_msg,
                    "duration_s": run_duration,
                })
            except Exception as e:
                logger.exception("Failed to commit candidate: %s", e)
                append_result_row({
                    "timestamp": datetime.utcnow().isoformat(),
                    "iteration": iteration,
                    "status": "commit_failed",
                    "bpb": bpb,
                    "commit": "",
                    "message": str(e),
                    "duration_s": run_duration,
                })
                # revert
                try:
                    repo.git.reset("--hard", "HEAD")
                except Exception:
                    pass
        else:
            # revert to last committed state
            try:
                repo.git.reset("--hard", "HEAD")
                logger.info("Candidate did not improve (bpb=%.6f) - reverted", bpb)
            except Exception as e:
                logger.exception("Revert failed: %s", e)
            append_result_row({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": iteration,
                "status": "not_improved",
                "bpb": bpb,
                "commit": "",
                "message": "not improved",
                "duration_s": run_duration,
            })

        # optional sleep between iterations
        if args.pause:
            time.sleep(args.pause)

        # end condition if time budget exceeded (optional)
        if args.max_seconds and (time.time() - args.start_time) > args.max_seconds:
            logger.info("Max seconds reached - stopping.")
            break

    logger.info("Main loop finished. Best BPB: %s", best_bpb if best_bpb < float("inf") else "n/a")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Agentless Autoresearch controller")
    p.add_argument("--backend", type=str, default="openai", help="LLM backend: openai (default)")
    p.add_argument("--model", type=str, default="gpt-5.4", help="LLM model name")
    p.add_argument("--run-timeout", type=int, default=DEFAULT_RUN_TIMEOUT, help="seconds per training run")
    p.add_argument("--max-iterations", type=int, default=100, help="max iterations")
    p.add_argument("--skip-baseline", action="store_true", help="skip baseline run")
    p.add_argument("--min-delta", type=float, default=1e-6, help="minimum improvement to consider")
    p.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between iterations")
    p.add_argument("--max-seconds", type=float, default=0.0, help="stop after N seconds (0 = disabled)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.start_time = time.time()
    # sanity checks
    if not os.path.exists(TARGET_FILE):
        print(f"ERROR: {TARGET_FILE} not found in repo root.")
        sys.exit(1)
    if not os.path.exists(PROMPT_FILE):
        logger.warning("%s not found - continuing without explicit instructions.", PROMPT_FILE)
    # Set up global vars for args
    args.max_iterations = int(args.max_iterations)
    # if using openai ensure OPENAI_API_KEY is set
    if args.backend.lower() == "openai" and not (os.environ.get("OPENAI_API_KEY") or openai.api_key):
        print("ERROR: OPENAI_API_KEY not set in environment.")
        sys.exit(1)
    # run main loop
    try:
        main_loop(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
