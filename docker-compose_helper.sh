#!/bin/bash
# docker-compose_helper.sh - Advanced runner for the Privateer Anomaly Detection stack.

# Exit immediately if a command fails.
set -e

# --- Default Argument Values ---
RUN_WITH_GPU=false
GPU_DEVICE_NUM=0
# ALVEO support is included structurally for future use.
RUN_WITH_ALVEO=false 

# --- Helper Functions ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script runs the multi-container application defined in docker-compose.yml."
    echo ""
    echo "Options:"
    echo "  --gpu <true|false>      Enable GPU support for the 'app' service (default: false)."
    echo "  --gpu_device <int>      Specify GPU device number to use (default: 0)."
    echo "  -h, --help              Show this help message."
    exit 1
}

validate_boolean() {
    local var_name="$1"
    local value="$2"
    if [[ "$value" != "true" && "$value" != "false" ]]; then
        echo "Error: Invalid value for $var_name. Must be 'true' or 'false'. Got: '$value'" >&2
        usage
    fi
}

validate_integer() {
    local var_name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: Invalid value for $var_name. Must be a non-negative integer. Got: '$value'" >&2
        usage
    fi
}

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) RUN_WITH_GPU_INPUT="${2,,}"; validate_boolean "--gpu" "$RUN_WITH_GPU_INPUT"; RUN_WITH_GPU="$RUN_WITH_GPU_INPUT"; shift ;;
        --gpu_device) GPU_DEVICE_NUM_INPUT="$2"; validate_integer "--gpu_device" "$GPU_DEVICE_NUM_INPUT"; GPU_DEVICE_NUM="$GPU_DEVICE_NUM_INPUT"; shift ;;
        --alveo) RUN_WITH_ALVEO_INPUT="${2,,}"; validate_boolean "--alveo" "$RUN_WITH_ALVEO_INPUT"; RUN_WITH_ALVEO="$RUN_WITH_ALVEO_INPUT"; shift ;; # Kept for future use
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# --- Load Environment Variables from .env ---
if [ -f .env ]; then
  # export makes them available to the shell and sub-processes like docker-compose
  export $(grep -v '^#' .env | xargs)
  echo "✅ Loaded environment variables from .env file."
else
  echo "⚠️ Warning: .env file not found. Some configurations may be missing." >&2
fi

# --- Initialize Docker Compose Parameters ---
BASE_COMPOSE_FILE="docker-compose.yml"
GPU_COMPOSE_FILE="docker-compose.gpu.yml"
COMPOSE_ARGS=("-f" "$BASE_COMPOSE_FILE")
ACTUAL_RUN_WITH_GPU=false

# --- GPU Specific Logic ---
if [ "$RUN_WITH_GPU" = "true" ]; then
    echo "INFO: GPU support requested."
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi -L | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            if [ "$GPU_DEVICE_NUM" -lt "$gpu_count" ]; then
                ACTUAL_RUN_WITH_GPU=true
                # EXPORT the variable that docker-compose.gpu.yml will use
                export NVIDIA_VISIBLE_DEVICES=$GPU_DEVICE_NUM
                COMPOSE_ARGS+=("-f" "$GPU_COMPOSE_FILE")
                gpu_name=$(nvidia-smi -L | sed -n "$((GPU_DEVICE_NUM + 1))p" | cut -d ':' -f2 | cut -d '(' -f1 | xargs)
                echo "Enabling GPU support. Using device $GPU_DEVICE_NUM: $gpu_name"
            else
                echo "Warning: GPU device number '$GPU_DEVICE_NUM' is out of range (0 to $((gpu_count - 1))). Running in CPU mode." >&2
            fi
        else
            echo "Warning: No NVIDIA GPUs found by nvidia-smi. Running in CPU mode." >&2
        fi
    else
        echo "Warning: nvidia-smi command not found. Cannot use GPU. Running in CPU mode." >&2
    fi
else
    echo "INFO: Running in CPU-only mode."
fi

# --- Final Command Execution ---
echo ""
echo "--- Docker Compose Configuration ---"
echo "GPU Support:          $RUN_WITH_GPU (Actual: $ACTUAL_RUN_WITH_GPU)"
if [ "$ACTUAL_RUN_WITH_GPU" = "true" ]; then
    echo "GPU Device:           $GPU_DEVICE_NUM ($gpu_name)"
fi
echo "Compose Files Used:   ${COMPOSE_ARGS[@]}"
echo "---"

echo "Pulling latest images and starting services..."
docker-compose "${COMPOSE_ARGS[@]}" up -d

echo ""
echo "Services are up and running."
echo "-----------------------------------"
echo "  - MLflow UI:      http://localhost:${PRIVATEER_MLFLOW_SERVER_PORT}"
echo "  - App Container:  'docker exec -it privateer-ad-gpu-app /bin/bash' to access"
echo "-----------------------------------"
echo ""
echo "To view logs, run: 'docker-compose ${COMPOSE_ARGS[@]} logs -f'"
echo "To stop services, run: 'docker-compose ${COMPOSE_ARGS[@]} down'"

exit 0