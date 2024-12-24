import os
from pathlib import Path
import subprocess
import tempfile
from typing import Optional, List

from executable import check_concorde_executable
from problem import Problem
from solution import Solution


def run_concorde(problem, concorde="concorde", extra_args=None):
    extra_args = extra_args or []
    check_concorde_executable(concorde)

    with tempfile.TemporaryDirectory() as tmp:
        tsp_fname = os.path.join(tmp, "problem.tsp")
        problem.to_tsp(tsp_fname)

        cmd = [concorde] + extra_args + [tsp_fname]
        subprocess.run(cmd, cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        sol_fname = os.path.join(tmp, "problem.sol")
        solution = Solution.from_file(sol_fname)

    return solution
