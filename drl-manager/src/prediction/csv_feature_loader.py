"""
CSV Feature Loader for Wind Power Prediction

Loads complete 13-feature historical data from turbine CSV files,
aligned with CloudSim Plus simulation time.

Time Alignment (COMPRESSED mode):
- Java GreenEnergyProvider skips first 12 CSV rows
- simulation_step=0 → CSV row 12
- simulation_step=1 → CSV row 13
- Python: actual_csv_row = simulation_step + 12
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CSVFeatureLoader:
    """
    Loads wind turbine features from CSV files with time alignment.

    Time Alignment:
    - CloudSim COMPRESSED mode: 1 simulation step = 1 second
    - CSV data: 1 row = 600 seconds (10 minutes)
    - Mapping: CSV_row_index = floor(simulation_time / 600)
    """

    def __init__(
        self,
        turbine_csv_paths: Dict[int, str],
        csv_timestep_seconds: int = 600,
        feature_columns: Optional[List[str]] = None,
        csv_start_offset: int = 12
    ):
        """
        Initialize the CSV feature loader.

        Args:
            turbine_csv_paths: Dict mapping turbine_id → CSV file path
                               e.g., {1: "path/to/turbine_001.csv", ...}
            csv_timestep_seconds: Time interval between CSV rows (default: 600s = 10min)
            feature_columns: List of feature column names (default: 13 features)
            csv_start_offset: Starting row offset in CSV (default: 12)
                             simulation_step=0 → CSV row 12
                             Ensures sufficient lookback history
        """
        self.turbine_csv_paths = turbine_csv_paths
        self.csv_timestep_seconds = csv_timestep_seconds
        self.csv_start_offset = csv_start_offset

        # Default 13-feature columns
        if feature_columns is None:
            self.feature_columns = [
                'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir',
                'Pab1', 'Prtv', 'T2m',
                'Sp', 'RelH', 'Wspd_w', 'Wdir_w',
                'Patv'
            ]
        else:
            self.feature_columns = feature_columns

        # Load and cache CSV data
        self.turbine_data: Dict[int, pd.DataFrame] = {}
        self._load_all_csvs()

        logger.info(
            f"CSVFeatureLoader initialized: {len(turbine_csv_paths)} turbines, "
            f"timestep={csv_timestep_seconds}s, features={len(self.feature_columns)}"
        )

    def _load_all_csvs(self):
        """Load all turbine CSV files into memory."""
        for turbine_id, csv_path in self.turbine_csv_paths.items():
            csv_path = Path(csv_path)

            if not csv_path.exists():
                logger.error(f"CSV file not found for turbine {turbine_id}: {csv_path}")
                continue

            try:
                # Load CSV
                df = pd.read_csv(csv_path)

                # Validate columns
                missing_cols = set(self.feature_columns) - set(df.columns)
                if missing_cols:
                    logger.warning(
                        f"Turbine {turbine_id}: Missing columns {missing_cols}, "
                        f"will fill with 0"
                    )
                    for col in missing_cols:
                        df[col] = 0.0

                # Select only needed columns
                df = df[self.feature_columns].copy()

                # Handle missing values
                df.fillna(0.0, inplace=True)

                self.turbine_data[turbine_id] = df

                logger.info(
                    f"Loaded turbine {turbine_id}: {len(df)} rows, "
                    f"{df.shape[1]} features from {csv_path.name}"
                )

            except Exception as e:
                logger.error(f"Failed to load CSV for turbine {turbine_id}: {e}")

    def sim_time_to_csv_index(self, simulation_time: float) -> int:
        """
        Convert simulation time to CSV row index.

        Args:
            simulation_time: Simulation time in seconds (actually simulation_step in COMPRESSED mode)

        Returns:
            CSV row index

        Example (COMPRESSED mode):
            simulation_step=0 → CSV row 12 (Java skips first 12 rows)
            simulation_step=1 → CSV row 13
            simulation_step=2 → CSV row 14
        """
        # In COMPRESSED mode: 1 simulation_step = 1 second = 1 CSV row
        # Java skips first 12 rows, so we add offset
        simulation_step = int(simulation_time)
        actual_csv_row = simulation_step + self.csv_start_offset
        return actual_csv_row

    def get_historical_features(
        self,
        turbine_id: int,
        current_sim_time: float,
        lookback_steps: int = 12
    ) -> Optional[np.ndarray]:
        """
        Get historical features for a turbine.

        Args:
            turbine_id: Turbine ID
            current_sim_time: Current simulation time in seconds
            lookback_steps: Number of historical timesteps to retrieve

        Returns:
            Array of shape (lookback_steps, num_features), or None if insufficient data
        """
        if turbine_id not in self.turbine_data:
            logger.warning(f"No data loaded for turbine {turbine_id}")
            return None

        df = self.turbine_data[turbine_id]

        # Get current CSV index
        current_idx = self.sim_time_to_csv_index(current_sim_time)

        # Calculate start index
        start_idx = current_idx - lookback_steps + 1

        # Check bounds
        if start_idx < 0:
            logger.warning(
                f"Turbine {turbine_id}: Insufficient history at sim_time={current_sim_time}s "
                f"(need {lookback_steps} steps, but start_idx={start_idx})"
            )
            return None

        if current_idx >= len(df):
            logger.warning(
                f"Turbine {turbine_id}: sim_time={current_sim_time}s exceeds CSV data "
                f"(current_idx={current_idx}, csv_length={len(df)})"
            )
            return None

        # Extract features
        features = df.iloc[start_idx:current_idx + 1].values  # shape: (lookback_steps, num_features)

        if features.shape[0] != lookback_steps:
            logger.warning(
                f"Turbine {turbine_id}: Retrieved {features.shape[0]} steps, "
                f"expected {lookback_steps}"
            )
            return None

        return features

    def get_feature_at_time(
        self,
        turbine_id: int,
        sim_time: float
    ) -> Optional[np.ndarray]:
        """
        Get features at a specific simulation time.

        Args:
            turbine_id: Turbine ID
            sim_time: Simulation time in seconds

        Returns:
            Array of shape (num_features,), or None if not available
        """
        if turbine_id not in self.turbine_data:
            return None

        df = self.turbine_data[turbine_id]
        idx = self.sim_time_to_csv_index(sim_time)

        if idx < 0 or idx >= len(df):
            return None

        return df.iloc[idx].values

    def get_data_range(self, turbine_id: int) -> Optional[Dict[str, float]]:
        """
        Get the simulation time range covered by CSV data.

        Args:
            turbine_id: Turbine ID

        Returns:
            Dict with 'min_sim_time', 'max_sim_time', 'num_rows'
        """
        if turbine_id not in self.turbine_data:
            return None

        df = self.turbine_data[turbine_id]
        num_rows = len(df)

        return {
            'min_sim_time': 0.0,
            'max_sim_time': (num_rows - 1) * self.csv_timestep_seconds,
            'num_rows': num_rows,
            'csv_timestep_seconds': self.csv_timestep_seconds
        }

    def validate_alignment(
        self,
        turbine_id: int,
        sim_time: float,
        actual_power_w: float,
        tolerance: float = 10000.0
    ) -> bool:
        """
        Validate that simulation power matches CSV data.

        Args:
            turbine_id: Turbine ID
            sim_time: Simulation time
            actual_power_w: Actual power from simulation (Watts)
            tolerance: Acceptable difference (Watts)

        Returns:
            True if aligned within tolerance
        """
        features = self.get_feature_at_time(turbine_id, sim_time)

        if features is None:
            return False

        # Patv is at index 12
        csv_power_kw = features[12]  # Patv in kW
        csv_power_w = csv_power_kw * 1000.0

        diff = abs(actual_power_w - csv_power_w)

        if diff > tolerance:
            logger.warning(
                f"Turbine {turbine_id} at sim_time={sim_time}s: "
                f"Power mismatch! Simulation={actual_power_w:.2f}W, "
                f"CSV={csv_power_w:.2f}W, diff={diff:.2f}W"
            )
            return False

        return True
