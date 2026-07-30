#!/bin/bash
set -uo pipefail

# Copyright (c) 2018-2026, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
        cat <<'EOF'
Usage: scripts/run_prepare_models.sh [MODEL_DIR ...]

Runs prepare_model.py in each model directory, creating a temporary pyenv-backed
virtual environment for each run and removing it afterward.

Examples:
    scripts/run_prepare_models.sh
    scripts/run_prepare_models.sh models/vision/detection/YOLO11 models/vision/detection/YOLOX
EOF
}

ensure_pyenv_python() {
    if ! command -v pyenv >/dev/null 2>&1; then
        echo "Error: pyenv is required" >&2
        exit 1
    fi

    if ! pyenv commands | grep -qx 'virtualenv'; then
        echo "Error: pyenv-virtualenv is required" >&2
        exit 1
    fi

}

collect_model_dirs() {
    if [[ $# -gt 0 ]]; then
        printf '%s\n' "$@"
        return
    fi

    find "$REPO_ROOT/models" -type f -name 'prepare_model.py' -print \
        | xargs -r -n1 dirname \
        | sort -u
}

run_prepare_model() {
    local model_dir="$1"
    local env_name=""
    local model_name=""

    (
        model_name="$(basename "$model_dir" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
        env_name="edgeai_${model_name}_$RANDOM"
        trap 'PYENV_VERSION="" pyenv uninstall -f "$env_name" >/dev/null 2>&1 || true' EXIT

        pyenv virtualenv "$env_name" >/dev/null
        export PYENV_VERSION="$env_name"

        cd "$model_dir"
        python prepare_model.py
    )
}

main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    local -a model_dirs=()
    mapfile -t model_dirs < <(collect_model_dirs "$@")

    if [[ ${#model_dirs[@]} -eq 0 ]]; then
        echo "No model directories found" >&2
        exit 1
    fi

    ensure_pyenv_python

    local ok=0 fail=0 skip=0
    local i=1 total=${#model_dirs[@]}

    for model_dir in "${model_dirs[@]}"; do
        echo "[$i/$total] $model_dir"

        if [[ ! -d "$model_dir" ]]; then
            echo "  [SKIP]"
            ((skip++))
            continue
        fi

        if [[ ! -f "$model_dir/prepare_model.py" ]]; then
            echo "  [SKIP]"
            ((skip++))
            continue
        fi

        model_dir="$(cd "$model_dir" && pwd)"

        if run_prepare_model "$model_dir"; then
            echo "  [OK]"
            ((ok++))
        else
            echo "  [FAIL]"
            ((fail++))
        fi
        echo

        ((i++))
    done

    echo "Done: $ok OK, $fail FAIL, $skip SKIP"
}

main "$@"

