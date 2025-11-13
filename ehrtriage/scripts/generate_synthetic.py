"""
Script to generate synthetic EHR data.

Usage:
    python -m ehrtriage.scripts.generate_synthetic
"""

import yaml
from ehrtriage.config import CONFIG_DIR
from ehrtriage.synthetic_data import generate_synthetic_data


def main():
    """Generate synthetic data with default configuration."""
    print("=" * 60)
    print("EHR Timeline Triage - Synthetic Data Generation")
    print("=" * 60)
    print()

    # Load config
    data_config_path = CONFIG_DIR / "data.yml"
    if data_config_path.exists():
        with open(data_config_path, "r") as f:
            data_config = yaml.safe_load(f)
        gen_config = data_config.get("synthetic_generation", {})
        print(f"Loaded gen_config: {gen_config}")
    else:
        gen_config = {}
        print("Config file not found")

    generate_synthetic_data(
        n_patients=gen_config.get("n_patients", 1000),
        n_admissions_range=tuple(gen_config.get("n_admissions_per_patient", [1, 3])),
        icu_rate=gen_config.get("icu_admission_rate", 0.3),
        readmission_rate=gen_config.get("readmission_rate", 0.15),
        mortality_rate=gen_config.get("mortality_rate", 0.10),
        event_freq_hours=tuple(gen_config.get("event_frequency_hours", [1, 4])),
        random_state=gen_config.get("random_state", 42),
    )

    print()
    print("=" * 60)
    print("Synthetic data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
