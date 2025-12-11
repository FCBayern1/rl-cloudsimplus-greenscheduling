"""
Tianshou Multi-Agent Training Entrypoint

Simplified entry point for running multi-agent hierarchical training
with Tianshou and PettingZoo ParallelEnv.

Features:
    - A2C and PPO algorithm support
    - Same multi-agent architecture as RLlib (Global + 10 Local agents)
    - TensorBoard logging
    - Checkpoint management
    - Action masking for local agents

Usage:
    # Method 1: Direct run (uses defaults from config.yml)
    python entrypoint_tianshou.py

    # Method 2: With environment variables
    export EXPERIMENT_ID="experiment_multi_dc_simple_tianshou"
    export ALGORITHM="A2C"
    python entrypoint_tianshou.py

    # Method 3: With command line arguments
    python entrypoint_tianshou.py --experiment experiment_multi_dc_simple_tianshou --algorithm A2C
"""

import os
# Set environment variables for compatibility
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
    """Main entry point for Tianshou multi-agent training."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Tianshou Multi-Agent Training for Multi-Datacenter MARL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from config.yml (A2C algorithm)
  python entrypoint_tianshou.py

  # Specify experiment
  python entrypoint_tianshou.py --experiment experiment_multi_dc_simple_tianshou

  # Use PPO instead of A2C
  python entrypoint_tianshou.py --algorithm PPO

  # Full customization
  python entrypoint_tianshou.py \\
      --experiment experiment_multi_dc_simple_tianshou \\
      --algorithm A2C \\
      --total-timesteps 100000 \\
      --device cuda
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
        help='Experiment ID from config.yml (default: experiment_multi_dc_simple_tianshou)'
    )

    # Training parameters
    parser.add_argument(
        '--algorithm',
        type=str,
        default=None,
        choices=['A2C', 'PPO'],
        help='RL algorithm to use (default: A2C)'
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=None,
        help='Total training timesteps (default: from config or 100000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'auto'],
        help='Device to use (default: auto)'
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
    print("  Tianshou Multi-Agent Training for Multi-Datacenter MARL")
    print("  Algorithms: A2C, PPO")
    print("=" * 70)
    print()

    # Resolve configuration path
    if args.config is None:
        args.config = os.getenv('CONFIG_FILE')
        if args.config is None:
            script_dir = Path(__file__).parent
            args.config = str(script_dir.parent / 'config.yml')

    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please specify --config or ensure ../config.yml exists")
        sys.exit(1)

    logger.info(f"Using configuration: {args.config}")

    # Resolve experiment ID
    if args.experiment is None:
        args.experiment = os.getenv('EXPERIMENT_ID', 'experiment_multi_dc_simple_tianshou')

    logger.info(f"Experiment: {args.experiment}")

    # Resolve algorithm
    if args.algorithm is None:
        args.algorithm = os.getenv('ALGORITHM')

    if args.algorithm:
        logger.info(f"Algorithm override: {args.algorithm}")

    # Resolve total_timesteps
    if args.total_timesteps is None:
        env_timesteps = os.getenv('TOTAL_TIMESTEPS')
        args.total_timesteps = int(env_timesteps) if env_timesteps else None

    if args.total_timesteps:
        logger.info(f"Total timesteps: {args.total_timesteps}")

    # Resolve device
    if args.device is None:
        args.device = os.getenv('DEVICE')

    if args.device:
        logger.info(f"Device: {args.device}")

    print()

    # Training mode
    logger.info("=" * 70)
    logger.info("Starting Tianshou Multi-Agent Training...")
    logger.info("=" * 70)
    print()

    # Import training script
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.training.train_tianshou_multidc import train_tianshou, load_config
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.error("Please ensure you are in the drl-manager directory")
        logger.error("And that Tianshou is installed: pip install tianshou")
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
    env_config = exp_config
    global_model_config = exp_config.get('global_model', {})
    local_model_config = exp_config.get('local_model', {})
    training_config = exp_config.get('training', {})

    # Apply command-line overrides
    if args.algorithm:
        training_config['algorithm'] = args.algorithm

    if args.total_timesteps:
        training_config['total_timesteps'] = args.total_timesteps

    if args.device:
        training_config['device'] = args.device

    # Determine algorithm name
    algorithm_name = training_config.get('algorithm', 'A2C').upper()

    # Resolve output directory
    if args.output_dir is None:
        env_output = os.getenv('OUTPUT_DIR')
        if env_output:
            args.output_dir = env_output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"../logs/{args.experiment}_{algorithm_name}/{timestamp}"

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_path / "training.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    # Save experiment configuration
    import yaml
    config_save_path = output_path / "experiment_config.yml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'experiment_name': args.experiment,
            'timestamp': datetime.now().isoformat(),
            'framework': 'tianshou',
            'command_line_args': {
                'config': args.config,
                'experiment': args.experiment,
                'algorithm': args.algorithm,
                'total_timesteps': args.total_timesteps,
                'device': args.device,
                'output_dir': args.output_dir,
            },
            'env_config': env_config,
            'global_model_config': global_model_config,
            'local_model_config': local_model_config,
            'training_config': training_config,
        }, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"Saved experiment config to: {config_save_path}")

    # Print configuration summary
    logger.info("=" * 70)
    logger.info("Configuration Summary:")
    logger.info("=" * 70)
    logger.info(f"  Framework: Tianshou")
    logger.info(f"  Algorithm: {algorithm_name}")
    logger.info(f"  Experiment: {args.experiment}")
    logger.info(f"  Multi-DC enabled: {env_config.get('multi_datacenter_enabled', False)}")
    logger.info(f"  Number of datacenters: {len(env_config.get('datacenters', []))}")
    logger.info(f"  Total timesteps: {training_config.get('total_timesteps', 100000)}")
    logger.info(f"  Device: {training_config.get('device', 'auto')}")
    logger.info(f"  Output dir: {args.output_dir}")

    # Print datacenter summary
    datacenters = env_config.get('datacenters', [])
    if datacenters:
        logger.info("")
        logger.info("  Datacenters:")
        for dc in datacenters:
            dc_name = dc.get('name', f"DC_{dc.get('datacenter_id', '?')}")
            turbine_ids = dc.get('turbine_ids', [dc.get('turbine_id', '?')])
            wind_data_file = dc.get('wind_data_file', '')

            is_solar = isinstance(wind_data_file, str) and "solarProduction" in wind_data_file

            if is_solar:
                logger.info(
                    f"    - {dc_name}: solar_energy={dc.get('green_energy_enabled', False)}"
                )
            else:
                logger.info(
                    f"    - {dc_name}: turbines={turbine_ids}, "
                    f"green_energy={dc.get('green_energy_enabled', False)}"
                )

    logger.info("=" * 70)
    print()

    # Check Java Gateway connection
    try:
        from py4j.java_gateway import JavaGateway, GatewayParameters
        gateway_port = env_config.get('py4j_port', 25333)
        logger.info(f"Checking Java Gateway connection on port {gateway_port}...")

        try:
            test_gateway = JavaGateway(
                gateway_parameters=GatewayParameters(port=gateway_port, auto_convert=True)
            )
            _ = test_gateway.entry_point
            test_gateway.close()
            logger.info("Java Gateway connection successful")
        except Exception as e:
            logger.warning("Cannot connect to Java Gateway")
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
        train_tianshou(
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
        logger.info("")
        logger.info("Saved files:")
        logger.info(f"  - experiment_config.yml  : Full experiment configuration")
        logger.info(f"  - training.log           : Complete training logs")
        logger.info(f"  - tensorboard/           : TensorBoard logs")
        logger.info(f"  - checkpoints/           : Model checkpoints")
        logger.info(f"  - best_model.pt          : Best model checkpoint")
        logger.info(f"  - final_model.pt         : Final model checkpoint")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. View TensorBoard: tensorboard --logdir={args.output_dir}/tensorboard")
        logger.info("  2. Load checkpoints for evaluation")
        logger.info("  3. Compare with RLlib PPO results")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
