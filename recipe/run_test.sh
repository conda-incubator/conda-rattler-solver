#!/bin/bash
set -euo pipefail

CONDA_SOLVER=rattler conda create --dry-run scipy
