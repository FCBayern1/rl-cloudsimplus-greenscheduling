#!/bin/bash
# ==============================================================================
# Docker Entrypoint Script
# ==============================================================================
# This script handles different startup modes for the container.
#
# Usage:
#   docker run ... rl-multidc train              # Start training
#   docker run ... rl-multidc tensorboard        # Start TensorBoard
#   docker run ... rl-multidc bash               # Interactive shell
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  RL Multi-Datacenter Training Environment${NC}"
echo -e "${GREEN}======================================================================${NC}"

# Function to print colored messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verify CUDA availability
log_info "Verifying GPU availability..."
if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    log_info "GPU detected: ${GPU_NAME}"
else
    log_warn "No GPU detected. Training will use CPU only."
fi

# Display environment info
log_info "Environment Configuration:"
echo "  - Experiment: ${EXPERIMENT_ID}"
echo "  - Num Workers: ${NUM_WORKERS}"
echo "  - Num GPUs: ${NUM_GPUS}"
echo "  - Total Timesteps: ${TOTAL_TIMESTEPS}"
echo ""

# Change to workspace
cd /workspace

# Parse command
COMMAND="${1:-train}"

case "$COMMAND" in
    train)
        log_info "Starting training..."
        echo ""

        # Start Java Gateway in background
        log_info "Starting CloudSim Plus Java Gateway..."
        cd cloudsimplus-gateway
        ./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC > /workspace/logs/java-gateway.log 2>&1 &
        JAVA_PID=$!
        cd /workspace

        # Wait for Java Gateway to start
        log_info "Waiting for Java Gateway to initialize..."
        sleep 10

        # Check if Java Gateway is running
        if ! kill -0 $JAVA_PID 2>/dev/null; then
            log_error "Java Gateway failed to start. Check logs/java-gateway.log"
            exit 1
        fi

        log_info "Java Gateway started successfully (PID: $JAVA_PID)"

        # Start Python training
        log_info "Starting Python training..."
        cd drl-manager

        python3 entrypoint_pettingzoo.py \
            --experiment "${EXPERIMENT_ID}" \
            --num-workers "${NUM_WORKERS}" \
            --num-gpus "${NUM_GPUS}" \
            --total-timesteps "${TOTAL_TIMESTEPS}"

        # Cleanup
        log_info "Training completed. Cleaning up..."
        kill $JAVA_PID 2>/dev/null || true
        ;;

    tensorboard)
        log_info "Starting TensorBoard..."
        tensorboard --logdir=/workspace/tensorboard --host=0.0.0.0 --port=6006
        ;;

    test)
        log_info "Running environment tests..."
        cd drl-manager
        python3 tests/test_pettingzoo_wind_prediction.py
        ;;

    bash|sh)
        log_info "Starting interactive shell..."
        exec /bin/bash
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        echo ""
        echo "Available commands:"
        echo "  train       - Start training (default)"
        echo "  tensorboard - Start TensorBoard server"
        echo "  test        - Run environment tests"
        echo "  bash        - Interactive shell"
        exit 1
        ;;
esac
