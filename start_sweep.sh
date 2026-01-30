#!/bin/bash

# Exit on error
set -e

# Arguments
if [ -f "$1" ]; then
    # First argument is the config file
    SWEEP_CONFIG="$1"
    NUMBER_JOBS="auto"
    SERVER_HOST=${2:-"nzucchet@euler.ethz.ch"}
    WORKSPACE_DIR=${3:-"Transformer-filters"}
    WANDB_PROJECT=${4:-"Transformer-filters"}
else
    # First argument is number of jobs (or auto)
    NUMBER_JOBS="$1"
    SWEEP_CONFIG="$2"
    SERVER_HOST=${3:-"nzucchet@euler.ethz.ch"}
    WORKSPACE_DIR=${4:-"Transformer-filters"}
    WANDB_PROJECT=${5:-"Transformer-filters"}
fi

# Check arguments
if [ -z "$NUMBER_JOBS" ] || [ -z "$SWEEP_CONFIG" ]; then
    echo "Usage: $0 [NUMBER_JOBS|auto] <SWEEP_CONFIG_FILE> [SERVER_HOST] [WORKSPACE_DIR] [WANDB_PROJECT]"
    echo "   or: $0 <SWEEP_CONFIG_FILE> [SERVER_HOST] [WORKSPACE_DIR] [WANDB_PROJECT] (auto-detects jobs)"
    exit 1
fi

# Check if config file exists
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: Configuration file '$SWEEP_CONFIG' not found."
    exit 1
fi

# Auto-detect number of jobs if requested
if [ "$NUMBER_JOBS" = "auto" ]; then
    echo "Calculating number of jobs from grid search config..."
    cat <<EOF > _calc_jobs.py
import yaml
import sys

try:
    with open('$SWEEP_CONFIG', 'r') as f:
        config = yaml.safe_load(f)

    if config.get('method') != 'grid':
        print('Error: Auto job count only supported for grid search. Please specify number of jobs manually.', file=sys.stderr)
        sys.exit(1)

    total = 1
    params = config.get('parameters', {})
    for p, v in params.items():
        if 'values' in v:
            total *= len(v['values'])
    print(total)
except Exception as e:
    print(f'Error calculating jobs: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    NUMBER_JOBS=$(uv run python _calc_jobs.py)
    rm _calc_jobs.py
    echo "Auto-detected number of jobs: $NUMBER_JOBS"
fi

# 1. Start wandb sweep locally
echo "Initializing sweep from $SWEEP_CONFIG..."
# Use uv run to ensure wandb is available and environment is correct
SWEEP_OUTPUT=$(uv run wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract Sweep ID
# Try to extract using the pattern from start_sweep.sh
SWEEP_CMD=$(echo "$SWEEP_OUTPUT" | grep -o "wandb agent [^ ]*")

if [ -n "$SWEEP_CMD" ]; then
    SWEEP_ID=$(echo "$SWEEP_CMD" | awk '{print $3}')
else
    # Fallback to the snippet's method
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/Creating sweep with ID:/ {print $NF}')
fi

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to extract sweep ID from wandb output."
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"

# 2. Connect to server, update code, and launch jobs
echo "Launching $NUMBER_JOBS agents on $SERVER_HOST for sweep $SWEEP_ID..."

# Create the job script locally
cat << EOF > agent_job.sh
#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --output=wandb_agent_%A_%a.out
#SBATCH --error=wandb_agent_%A_%a.err

# Ensure we are in the workspace
cd ~/$WORKSPACE_DIR

source .venv/bin/activate

# Dynamic LD_LIBRARY_PATH setup for JAX/NVIDIA
export LD_LIBRARY_PATH=\$(python -c 'import os, glob, sysconfig; print(":".join(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia/*/lib"))))'):\$LD_LIBRARY_PATH

wandb agent --count 1 --project $WANDB_PROJECT $SWEEP_ID
EOF

# Copy script to server
echo "Copying job script to $SERVER_HOST..."
scp agent_job.sh $SERVER_HOST:~/$WORKSPACE_DIR/
rm agent_job.sh

# Construct the remote command
# 1. Go to workspace
# 2. Pull latest changes
# 3. Activate environment (assuming .venv)
# 4. Submit Slurm job array
REMOTE_CMD="cd ~/$WORKSPACE_DIR && \
    git pull && \
    export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\" && \
    (command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh) && \
    uv sync && \
    sbatch --array=1-${NUMBER_JOBS} --job-name='${SWEEP_CONFIG##*/}' agent_job.sh"

echo "Executing remote command on $SERVER_HOST..."
echo "REMINDER: Please ensure you have committed and pushed your latest changes (including uv.lock) to the repository!"
ssh "$SERVER_HOST" "$REMOTE_CMD"

echo "Jobs submitted on $SERVER_HOST."

