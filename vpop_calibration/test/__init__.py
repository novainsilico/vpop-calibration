from vpop_calibration.config import smoke_test

import subprocess
import os

print(f"Initializing test pipeline. {smoke_test=}")

# Build simwork executable once for all tests
build_result = subprocess.run(
    [
        "nix",
        "build",
        ".#simwork.legacyPackages.x86_64-linux.perf.scripts.run-model-simple",
        "--print-out-paths",
    ],
    capture_output=True,
)
if build_result.returncode != 0:
    raise RuntimeError(build_result.stderr)
simwork_executable_tests = (
    build_result.stdout.decode().strip("\n") + "/bin/scripts.run-model-simple"
)
os.environ["SIMWORK_EXE"] = simwork_executable_tests
