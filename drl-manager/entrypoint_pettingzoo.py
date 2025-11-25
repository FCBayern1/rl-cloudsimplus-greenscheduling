"""
PettingZoo Parallel Training Entrypoint

Simplified entry point for running multi-agent hierarchical training
with PettingZoo ParallelEnv and RLlib.

Usage:
    # Method 1: Direct run (uses defaults from config.yml)
    python entrypoint_pettingzoo.py

    # Method 2: With environment variables
    export EXPERIMENT_ID="experiment_multi_dc_3"
    export NUM_WORKERS=4
    export TOTAL_TIMESTEPS=100000
    python entrypoint_pettingzoo.py

    # Method 3: With command line arguments
    python entrypoint_pettingzoo.py --experiment experiment_multi_dc_3 --num-workers 4

Features:
    - Auto-configuration from config.yml
    - Wind prediction integration (if enabled in config)
    - RLlib distributed training
    - TensorBoard logging
    - Checkpoint management
"""

import os
# Set environment variables for Windows compatibility
# Must be set before importing numpy/torch/ray
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for PettingZoo parallel training."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="PettingZoo Parallel Training for Multi-Datacenter MARL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from config.yml
  python entrypoint_pettingzoo.py

  # Specify experiment
  python entrypoint_pettingzoo.py --experiment experiment_multi_dc_3

  # Full customization
  python entrypoint_pettingzoo.py \\
      --experiment experiment_multi_dc_3 \\
      --num-workers 8 \\
      --total-timesteps 200000 \\
      --num-gpus 1

  # With environment variables (PowerShell)
  $env:EXPERIMENT_ID = "experiment_multi_dc_3"
  $env:NUM_WORKERS = "4"
  python entrypoint_pettingzoo.py
        """
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yml (default: ../config.yml)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment ID from config.yml (default: from env var or experiment_multi_dc_3)'
    )

    # Training parameters
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: from env var or 4)'
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=None,
        help='Total training timesteps (default: from config or 100000)'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (default: 0)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for logs and checkpoints (default: auto-generated)'
    )

    # Misc
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test environment without training'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Enable verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print("=" * 70)
    print("  PettingZoo Parallel Training for Multi-Datacenter MARL")
    print("=" * 70)
    print()

    # Resolve configuration path
    if args.config is None:
        # Try environment variable
        args.config = os.getenv('CONFIG_FILE')
        if args.config is None:
            # Default to ../config.yml
            script_dir = Path(__file__).parent
            args.config = str(script_dir.parent / 'config.yml')

    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please specify --config or ensure ../config.yml exists")
        sys.exit(1)

    logger.info(f"Using configuration: {args.config}")

    # Resolve experiment ID
    if args.experiment is None:
        args.experiment = os.getenv('EXPERIMENT_ID', 'experiment_multi_dc_3')

    logger.info(f"Experiment: {args.experiment}")

    # Resolve num_workers
    if args.num_workers is None:
        env_workers = os.getenv('NUM_WORKERS')
        args.num_workers = int(env_workers) if env_workers else None  # Use None to respect config

    if args.num_workers is not None:
        logger.info(f"Number of workers: {args.num_workers}")

    # Resolve total_timesteps
    if args.total_timesteps is None:
        env_timesteps = os.getenv('TOTAL_TIMESTEPS')
        args.total_timesteps = int(env_timesteps) if env_timesteps else None

    if args.total_timesteps:
        logger.info(f"Total timesteps: {args.total_timesteps}")

    # Resolve num_gpus
    if args.num_gpus is None:
        env_gpus = os.getenv('NUM_GPUS')
        args.num_gpus = int(env_gpus) if env_gpus else None  # Keep None to use config default

    if args.num_gpus is not None:
        logger.info(f"Number of GPUs: {args.num_gpus}")

    # Resolve output directory (same structure as Stable Baselines3)
    if args.output_dir is None:
        env_output = os.getenv('OUTPUT_DIR')
        if env_output:
            args.output_dir = env_output
        else:
            # Auto-generate with timestamp: logs/experiment_name/timestamp/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"../logs/{args.experiment}/{timestamp}"

    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"  (Multiple runs of the same experiment will be grouped under logs/{args.experiment}/)")
    print()

    # Test mode - just verify environment
    if args.test:
        logger.info("=" * 70)
        logger.info("TEST MODE: Verifying environment setup...")
        logger.info("=" * 70)

        try:
            # Import test script
            sys.path.insert(0, str(Path(__file__).parent))
            from tests.test_pettingzoo_wind_prediction import test_pettingzoo_with_prediction

            success = test_pettingzoo_with_prediction()

            if success:
                logger.info("\n✓ Environment test PASSED")
                logger.info("You can now run full training by removing --test flag")
                sys.exit(0)
            else:
                logger.error("\n✗ Environment test FAILED")
                logger.error("Please fix the issues before running training")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            sys.exit(1)

    # Training mode
    logger.info("=" * 70)
    logger.info("Starting RLlib PettingZoo Parallel Training...")
    logger.info("=" * 70)
    print()

    # Import training script
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.training.train_rllib_multidc import train_rllib, load_config
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.error("Please ensure you are in the drl-manager directory")
        logger.error("And that all dependencies are installed: pip install -r requirements_rllib.txt")
        sys.exit(1)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}...")
    all_config = load_config(args.config)

    if args.experiment not in all_config:
        logger.error(f"Experiment '{args.experiment}' not found in {args.config}")
        logger.error(f"Available experiments: {list(all_config.keys())}")
        sys.exit(1)

    exp_config = all_config[args.experiment]

    # Extract sub-configurations
    env_config = exp_config  # PettingZoo uses full config
    global_model_config = exp_config.get('global_model', {})
    local_model_config = exp_config.get('local_model', {})
    training_config = exp_config.get('training', {})

    # Override with command line arguments
    if args.num_workers is not None:
        training_config['num_workers'] = args.num_workers

    if args.total_timesteps is not None:
        training_config['total_timesteps'] = args.total_timesteps

    if args.num_gpus is not None:
        training_config['num_gpus'] = args.num_gpus

    # Print configuration summary
    logger.info("=" * 70)
    logger.info("Configuration Summary:")
    logger.info("=" * 70)
    logger.info(f"  Experiment: {args.experiment}")
    logger.info(f"  Multi-DC enabled: {env_config.get('multi_datacenter_enabled', False)}")
    logger.info(f"  Number of datacenters: {len(env_config.get('datacenters', []))}")
    logger.info(f"  Wind prediction: {env_config.get('wind_prediction', {}).get('enabled', False)}")
    logger.info(f"  Num workers: {training_config.get('num_workers', 4)}")
    logger.info(f"  Total timesteps: {training_config.get('total_timesteps', 100000)}")
    logger.info(f"  Num GPUs: {training_config.get('num_gpus', 0)}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info("=" * 70)
    print()

    # Check Java Gateway connection (optional warning)
    try:
        from py4j.java_gateway import JavaGateway, GatewayParameters
        gateway_port = env_config.get('py4j_port', 25333)
        logger.info(f"Checking Java Gateway connection on port {gateway_port}...")

        # Try to connect (with short timeout)
        try:
            test_gateway = JavaGateway(
                gateway_parameters=GatewayParameters(port=gateway_port, auto_convert=True)
            )
            # Try to access entry point
            _ = test_gateway.entry_point
            test_gateway.close()
            logger.info("✓ Java Gateway connection successful")
        except Exception as e:
            logger.warning("✗ Cannot connect to Java Gateway")
            logger.warning(f"  Error: {e}")
            logger.warning(f"  Please ensure Java Gateway is running:")
            logger.warning(f"    cd cloudsimplus-gateway")
            logger.warning(f"    ./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC")
            logger.warning("")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Exiting...")
                sys.exit(1)
    except ImportError:
        logger.warning("py4j not imported, skipping gateway check")

    print()

    # Start training
    try:
        train_rllib(
            env_config=env_config,
            global_model_config=global_model_config,
            local_model_config=local_model_config,
            training_config=training_config,
            output_dir=args.output_dir
        )

        logger.info("\n" + "=" * 70)
        logger.info("Training completed successfully!")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. View training logs in the output directory")
        logger.info("  2. Load checkpoints for evaluation")
        logger.info("  3. Adjust hyperparameters in config.yml for better results")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
