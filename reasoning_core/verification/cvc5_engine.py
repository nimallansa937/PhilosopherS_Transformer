"""
CVC5 Parallel Verification Engine.

Runs CVC5 in parallel with Z3 on formal claims. Different solvers
have different strengths — running both catches cases where one
times out but the other solves instantly.

CVC5 handles certain quantifier patterns and bitvector theories
better than Z3. For our use case (modal logic, first-order philosophy),
Z3 is primary. CVC5 is insurance.

Falls back gracefully if cvc5 is not installed.

Reference: PHILOSOPHER_ENGINE_V3_UNIFIED_ARCHITECTURE.md, §3.2
"""

import subprocess
import tempfile
import os
import concurrent.futures
from typing import Optional


def run_z3(z3_code: str, timeout_s: float = 30.0) -> str:
    """Run Z3 code via subprocess and return result.

    Args:
        z3_code: Z3 Python script that prints 'sat', 'unsat', or 'unknown'
        timeout_s: Timeout in seconds

    Returns: 'sat', 'unsat', 'unknown', 'error', or 'timeout'
    """
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False) as f:
        f.write(z3_code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['python', tmp_path],
            capture_output=True, text=True,
            timeout=timeout_s,
        )
        output = result.stdout.strip().lower()
        if 'unsat' in output:
            return 'unsat'
        elif 'sat' in output:
            return 'sat'
        elif 'unknown' in output:
            return 'unknown'
        return 'error'
    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception:
        return 'error'
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_cvc5(smt2_code: str, timeout_s: float = 30.0) -> str:
    """Run CVC5 on SMT-LIB2 input via subprocess.

    Args:
        smt2_code: SMT-LIB2 formatted input
        timeout_s: Timeout in seconds

    Returns: 'sat', 'unsat', 'unknown', 'error', or 'timeout'
    """
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.smt2', delete=False) as f:
        f.write(smt2_code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['cvc5', '--tlimit-per', str(int(timeout_s * 1000)), tmp_path],
            capture_output=True, text=True,
            timeout=timeout_s + 5,
        )
        output = result.stdout.strip().lower()
        if 'unsat' in output:
            return 'unsat'
        elif 'sat' in output:
            return 'sat'
        elif 'unknown' in output:
            return 'unknown'
        return 'error'
    except FileNotFoundError:
        return 'error'  # cvc5 not installed
    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception:
        return 'error'
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def verify_parallel(z3_code: str, cvc5_code: Optional[str] = None,
                    timeout_s: float = 30.0) -> str:
    """Run Z3 and CVC5 in parallel. First definitive result wins.

    If cvc5_code is None, only Z3 is used (graceful fallback).

    Args:
        z3_code: Z3 Python code or SMT-LIB2 for Z3
        cvc5_code: SMT-LIB2 code for CVC5 (optional)
        timeout_s: Timeout per solver in seconds

    Returns: 'sat', 'unsat', 'unknown', 'timeout', or 'error'
    """
    if cvc5_code is None:
        # No CVC5 code provided — just run Z3
        return run_z3(z3_code, timeout_s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        z3_future = pool.submit(run_z3, z3_code, timeout_s)
        cvc5_future = pool.submit(run_cvc5, cvc5_code, timeout_s)

        done, pending = concurrent.futures.wait(
            [z3_future, cvc5_future],
            return_when=concurrent.futures.FIRST_COMPLETED,
            timeout=timeout_s,
        )

        # Check completed futures for definitive results
        for future in done:
            try:
                result = future.result(timeout=1)
                if result in ("sat", "unsat"):
                    # Cancel remaining
                    for p in pending:
                        p.cancel()
                    return result
            except Exception:
                continue

        # Wait for remaining futures
        for future in pending:
            try:
                result = future.result(timeout=timeout_s)
                if result in ("sat", "unsat"):
                    return result
            except Exception:
                continue

        return "timeout"


def is_cvc5_available() -> bool:
    """Check if CVC5 is installed and accessible."""
    try:
        result = subprocess.run(
            ['cvc5', '--version'],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
