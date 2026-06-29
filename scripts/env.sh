#!/usr/bin/env bash
# Shell environment for correlation-length-investigation (local or Della).
#
# Usage — source from your SSH session (recommended):
#   source /path/to/correlation-length-investigation/scripts/env.sh
#
# Optional overrides (before sourcing):
#   export PROJECT_ROOT=/custom/path
#   export LATTICE_GAS_ROOT=$HOME/software/lattice-gas
#   export FLEX_CONDA_ENV=lattice
#   export FLEX_ENV_CD=0          # skip cd into PROJECT_ROOT

_flex_env_setup() {
    local script_dir repo_root host

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
    host="$(hostname -s 2>/dev/null || hostname)"

    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        if [[ "${host}" == della* ]] || [[ "$(hostname -f 2>/dev/null)" == *della* ]]; then
            if [[ -d "/scratch/gpfs/WJACOBS/${USER}/correlation-length-investigation" ]]; then
                PROJECT_ROOT="/scratch/gpfs/WJACOBS/${USER}/correlation-length-investigation"
            else
                PROJECT_ROOT="${repo_root}"
            fi
        else
            PROJECT_ROOT="${repo_root}"
        fi
    fi
    export PROJECT_ROOT

    if [[ -z "${LATTICE_GAS_ROOT:-}" ]]; then
        LATTICE_GAS_ROOT="${HOME}/software/lattice-gas"
    fi
    export LATTICE_GAS_ROOT

    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    FLEX_CONDA_ENV="${FLEX_CONDA_ENV:-lattice}"
    if [[ "${host}" == della* ]] || [[ "$(hostname -f 2>/dev/null)" == *della* ]]; then
        if command -v module >/dev/null 2>&1; then
            module load anaconda3/2024.10 2>/dev/null || true
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${FLEX_CONDA_ENV}" 2>/dev/null || {
            echo "[env] WARNING: could not activate conda env '${FLEX_CONDA_ENV}'" >&2
        }
    else
        echo "[env] WARNING: conda not found — activate '${FLEX_CONDA_ENV}' manually" >&2
    fi

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        case "$(uname -s)" in
            Darwin)
                export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}"
                ;;
            *)
                export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
                ;;
        esac
    fi

    export PYTHONUNBUFFERED=1

    if [[ "${FLEX_ENV_CD:-1}" != "0" ]] && [[ -d "${PROJECT_ROOT}" ]]; then
        cd "${PROJECT_ROOT}" || true
    fi
}

_flex_env_status() {
    echo "correlation-length-investigation environment"
    echo "  host:            $(hostname -s 2>/dev/null || hostname)"
    echo "  PROJECT_ROOT:    ${PROJECT_ROOT:-<unset>}"
    echo "  LATTICE_GAS_ROOT:${LATTICE_GAS_ROOT:-<unset>}"
    echo "  PYTHONPATH:      ${PYTHONPATH:-<unset>}"
    echo "  CONDA_PREFIX:    ${CONDA_PREFIX:-<unset>}"
    echo "  pwd:             $(pwd)"
    if command -v python >/dev/null 2>&1; then
        python -c "
from lattice_gas.simulate import simulate
from coex.flex_coex import coex_chemical_potential
print('  import check:    OK')
" 2>/dev/null || echo "  import check:    FAILED (build lattice-gas or activate conda env)"
    else
        echo "  import check:    skipped (python not found)"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script is meant to be sourced, not executed:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    echo >&2
    _flex_env_setup
    _flex_env_status
    exit 0
fi

_flex_env_setup
_flex_env_status

unset -f _flex_env_setup _flex_env_status
