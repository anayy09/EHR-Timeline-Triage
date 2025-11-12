"""
Script to generate synthetic EHR data.

Usage:
    python -m ehrtriage.scripts.generate_synthetic
"""

from ehrtriage.synthetic_data import generate_synthetic_data


def main():
    """Generate synthetic data with default configuration."""
    print("=" * 60)
    print("EHR Timeline Triage - Synthetic Data Generation")
    print("=" * 60)
    print()

    generate_synthetic_data()

    print()
    print("=" * 60)
    print("Synthetic data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
