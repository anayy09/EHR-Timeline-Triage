"""
Synthetic EHR data generation.

Generates realistic synthetic patient data for demonstration purposes.
This allows the system to run without access to real PHI.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from ehrtriage.config import CONFIG_DIR, DATA_DIR


def generate_patients(n_patients: int, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic patient demographics.

    Args:
        n_patients: Number of patients to generate
        random_state: Random seed

    Returns:
        DataFrame with patient information
    """
    np.random.seed(random_state)

    patients = []
    for i in range(n_patients):
        subject_id = f"P{i:06d}"
        sex = np.random.choice(["M", "F"], p=[0.48, 0.52])
        birth_year = np.random.randint(1930, 2005)

        patients.append(
            {
                "subject_id": subject_id,
                "sex": sex,
                "birth_year": birth_year,
            }
        )

    return pd.DataFrame(patients)


def generate_admissions(
    patients_df: pd.DataFrame,
    n_admissions_range: Tuple[int, int] = (1, 3),
    readmission_rate: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hospital admissions.

    Args:
        patients_df: Patient demographics
        n_admissions_range: Range of admissions per patient
        readmission_rate: Probability of readmission
        random_state: Random seed

    Returns:
        DataFrame with admission information
    """
    np.random.seed(random_state)

    admissions = []
    hadm_counter = 0
    base_date = datetime(2020, 1, 1)

    for _, patient in patients_df.iterrows():
        subject_id = patient["subject_id"]
        age = 2024 - patient["birth_year"]

        # Number of admissions for this patient
        n_admissions = np.random.randint(*n_admissions_range)

        # Generate admissions with potential readmissions
        current_date = base_date + timedelta(days=np.random.randint(0, 365))

        for adm_idx in range(n_admissions):
            hadm_id = f"H{hadm_counter:08d}"
            hadm_counter += 1

            # Length of stay (skewed distribution)
            los_days = np.random.gamma(2, 2)  # Mean ~4 days
            los_days = np.clip(los_days, 1, 30)

            admittime = current_date
            dischtime = admittime + timedelta(days=los_days)

            # Determine if patient dies during admission (low probability)
            dies = np.random.random() < 0.02
            deathtime = (
                dischtime - timedelta(hours=np.random.randint(1, int(los_days * 24)))
                if dies
                else None
            )

            admissions.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "admittime": admittime,
                    "dischtime": dischtime if not dies else deathtime,
                    "deathtime": deathtime,
                    "age": age,
                }
            )

            # Schedule next admission (if not last and not died)
            if not dies and adm_idx < n_admissions - 1:
                # Readmission: could be early (30 days) or later
                if np.random.random() < readmission_rate:
                    days_to_next = np.random.randint(1, 30)
                else:
                    days_to_next = np.random.randint(31, 180)

                current_date = dischtime + timedelta(days=days_to_next)

    return pd.DataFrame(admissions)


def generate_icu_stays(
    admissions_df: pd.DataFrame,
    icu_rate: float = 0.3,
    mortality_rate: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic ICU stays from admissions.

    Args:
        admissions_df: Hospital admissions
        icu_rate: Fraction of admissions that have ICU stay
        mortality_rate: ICU mortality rate
        random_state: Random seed

    Returns:
        DataFrame with ICU stay information
    """
    np.random.seed(random_state)

    # Select admissions for ICU
    n_icu = int(len(admissions_df) * icu_rate)
    icu_admissions = admissions_df.sample(n=n_icu, random_state=random_state)

    icu_stays = []
    stay_counter = 0

    for _, adm in icu_admissions.iterrows():
        stay_id = f"ICU{stay_counter:08d}"
        stay_counter += 1

        # ICU admission slightly after hospital admission
        intime = adm["admittime"] + timedelta(hours=np.random.randint(0, 24))

        # ICU length of stay
        icu_los_days = np.random.gamma(2, 1.5)  # Mean ~3 days
        icu_los_days = np.clip(icu_los_days, 2, 20)  # At least 2 days (48h)

        outtime = intime + timedelta(days=icu_los_days)

        # Ensure ICU stay doesn't exceed hospital stay
        if outtime > adm["dischtime"]:
            outtime = adm["dischtime"]

        # Mortality in ICU
        if adm["deathtime"] is not None:
            # Patient died during admission - likely in ICU
            deathtime = adm["deathtime"]
        elif np.random.random() < mortality_rate:
            # New death during ICU stay (after 48h mark)
            hours_after_48 = np.random.randint(1, int((outtime - intime).total_seconds() / 3600) - 48)
            deathtime = intime + timedelta(hours=48 + hours_after_48)
        else:
            deathtime = None

        icu_stays.append(
            {
                "stay_id": stay_id,
                "subject_id": adm["subject_id"],
                "hadm_id": adm["hadm_id"],
                "intime": intime,
                "outtime": outtime,
                "deathtime": deathtime,
            }
        )

    return pd.DataFrame(icu_stays)


def generate_events(
    admissions_df: pd.DataFrame,
    icu_stays_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    event_freq_hours: Tuple[float, float] = (1, 4),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic clinical events (vitals, labs, meds).

    Args:
        admissions_df: Hospital admissions
        icu_stays_df: ICU stays
        patients_df: Patient demographics
        event_freq_hours: Range of hours between events
        random_state: Random seed

    Returns:
        DataFrame with clinical events
    """
    np.random.seed(random_state)

    events = []

    # Merge to get patient info
    admissions_with_patient = admissions_df.merge(
        patients_df[["subject_id", "sex"]], on="subject_id"
    )

    # Event type definitions
    vital_types = {
        "heart_rate": (60, 100, 15),
        "sbp": (110, 130, 15),
        "dbp": (70, 85, 10),
        "map": (70, 100, 12),
        "respiratory_rate": (12, 20, 4),
        "spo2": (95, 100, 3),
        "temperature": (36.5, 37.2, 0.5),
    }

    lab_types = {
        "lactate": (0.5, 2.0, 0.5),
        "creatinine": (0.7, 1.3, 0.3),
        "wbc": (4.5, 11.0, 2.0),
        "hemoglobin": (12.0, 16.0, 2.0),
        "platelet": (150, 400, 80),
        "sodium": (136, 145, 3),
        "potassium": (3.5, 5.0, 0.5),
        "bun": (7, 20, 5),
    }

    med_types = ["vasopressor", "antibiotic", "sedative", "diuretic", "anticoagulant"]

    for _, adm in admissions_with_patient.iterrows():
        subject_id = adm["subject_id"]
        hadm_id = adm["hadm_id"]
        start_time = adm["admittime"]
        end_time = adm["dischtime"]

        # Check if ICU stay
        icu_stay = icu_stays_df[icu_stays_df["hadm_id"] == hadm_id]
        is_icu = len(icu_stay) > 0
        will_die = adm["deathtime"] is not None

        # Generate events throughout stay
        current_time = start_time

        while current_time < end_time:
            # Vitals (more frequent)
            for vital_name, (mean, std_mean, std) in vital_types.items():
                # Abnormal values if patient is deteriorating
                if will_die and current_time > (end_time - timedelta(hours=24)):
                    if vital_name == "heart_rate":
                        value = np.random.normal(120, 20)
                    elif vital_name in ["sbp", "map"]:
                        value = np.random.normal(85, 15)
                    elif vital_name == "spo2":
                        value = np.random.normal(88, 8)
                    else:
                        value = np.random.normal(mean, std * 2)
                else:
                    value = np.random.normal(np.random.normal(mean, std_mean), std)

                events.append(
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "stay_id": icu_stay.iloc[0]["stay_id"] if is_icu else None,
                        "time": current_time,
                        "type": "vital",
                        "code": vital_name,
                        "value": float(value),
                    }
                )

            # Labs (less frequent)
            if np.random.random() < 0.3:
                for lab_name, (mean, std_mean, std) in lab_types.items():
                    # Abnormal labs if deteriorating
                    if will_die and current_time > (end_time - timedelta(hours=24)):
                        if lab_name == "lactate":
                            value = np.random.normal(5.0, 2.0)
                        elif lab_name == "creatinine":
                            value = np.random.normal(2.5, 1.0)
                        else:
                            value = np.random.normal(mean, std * 2)
                    else:
                        value = np.random.normal(np.random.normal(mean, std_mean), std)

                    events.append(
                        {
                            "subject_id": subject_id,
                            "hadm_id": hadm_id,
                            "stay_id": icu_stay.iloc[0]["stay_id"] if is_icu else None,
                            "time": current_time,
                            "type": "lab",
                            "code": lab_name,
                            "value": float(value),
                        }
                    )

            # Medications
            if is_icu and np.random.random() < 0.2:
                med = np.random.choice(med_types)
                events.append(
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "stay_id": icu_stay.iloc[0]["stay_id"],
                        "time": current_time,
                        "type": "medication",
                        "code": med,
                        "value": 1.0,
                    }
                )

            # Next time step
            current_time += timedelta(
                hours=np.random.uniform(*event_freq_hours)
            )

    return pd.DataFrame(events)


def generate_comorbidities(
    patients_df: pd.DataFrame, random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic comorbidity flags for patients.

    Args:
        patients_df: Patient demographics
        random_state: Random seed

    Returns:
        DataFrame with comorbidity flags added
    """
    np.random.seed(random_state)

    patients_df = patients_df.copy()

    # Age-based comorbidity rates
    for idx, row in patients_df.iterrows():
        age = 2024 - row["birth_year"]

        # Higher rates for older patients
        age_factor = min(age / 80.0, 1.0)

        patients_df.loc[idx, "comorbidity_chf"] = np.random.random() < (0.05 + 0.15 * age_factor)
        patients_df.loc[idx, "comorbidity_renal"] = np.random.random() < (0.03 + 0.10 * age_factor)
        patients_df.loc[idx, "comorbidity_liver"] = np.random.random() < 0.05
        patients_df.loc[idx, "comorbidity_copd"] = np.random.random() < (0.05 + 0.15 * age_factor)
        patients_df.loc[idx, "comorbidity_diabetes"] = np.random.random() < (0.08 + 0.12 * age_factor)

    # Convert to boolean
    comorbidity_cols = [
        "comorbidity_chf",
        "comorbidity_renal",
        "comorbidity_liver",
        "comorbidity_copd",
        "comorbidity_diabetes",
    ]
    for col in comorbidity_cols:
        patients_df[col] = patients_df[col].astype(bool)

    return patients_df


def generate_synthetic_data(
    n_patients: int = 1000,
    n_admissions_range: Tuple[int, int] = (1, 3),
    icu_rate: float = 0.3,
    readmission_rate: float = 0.15,
    mortality_rate: float = 0.10,
    event_freq_hours: Tuple[float, float] = (1, 4),
    random_state: int = 42,
    output_dir: Path = DATA_DIR / "synthetic",
) -> Dict[str, pd.DataFrame]:
    """
    Generate complete synthetic EHR dataset.

    Args:
        n_patients: Number of patients
        n_admissions_range: Range of admissions per patient
        icu_rate: Fraction with ICU stays
        readmission_rate: Readmission probability
        mortality_rate: ICU mortality rate
        event_freq_hours: Event frequency range
        random_state: Random seed
        output_dir: Output directory for files

    Returns:
        Dictionary of generated DataFrames
    """
    print("Generating synthetic EHR data...")
    print(f"  - Patients: {n_patients}")

    # Generate patients
    patients_df = generate_patients(n_patients, random_state)
    patients_df = generate_comorbidities(patients_df, random_state)

    # Generate admissions
    admissions_df = generate_admissions(
        patients_df, n_admissions_range, readmission_rate, random_state
    )
    print(f"  - Admissions: {len(admissions_df)}")

    # Generate ICU stays
    icu_stays_df = generate_icu_stays(
        admissions_df, icu_rate, mortality_rate, random_state
    )
    print(f"  - ICU stays: {len(icu_stays_df)}")

    # Update admissions with ICU deaths
    admissions_df = admissions_df.merge(
        icu_stays_df[["hadm_id", "deathtime"]].rename(columns={"deathtime": "icu_deathtime"}),
        on="hadm_id",
        how="left",
    )
    admissions_df["deathtime"] = admissions_df["deathtime"].fillna(
        admissions_df["icu_deathtime"]
    )
    admissions_df = admissions_df.drop(columns=["icu_deathtime"])

    # Generate events
    events_df = generate_events(
        admissions_df, icu_stays_df, patients_df, event_freq_hours, random_state
    )
    print(f"  - Events: {len(events_df)}")

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)

    patients_df.to_parquet(output_dir / "patients.parquet", index=False)
    admissions_df.to_parquet(output_dir / "admissions.parquet", index=False)
    icu_stays_df.to_parquet(output_dir / "icustays.parquet", index=False)
    events_df.to_parquet(output_dir / "events.parquet", index=False)

    print(f"\nSaved synthetic data to {output_dir}")

    return {
        "patients": patients_df,
        "admissions": admissions_df,
        "icustays": icu_stays_df,
        "events": events_df,
    }


if __name__ == "__main__":
    # Load config
    data_config_path = CONFIG_DIR / "data.yml"
    if data_config_path.exists():
        with open(data_config_path, "r") as f:
            data_config = yaml.safe_load(f)
        gen_config = data_config.get("synthetic_generation", {})
    else:
        gen_config = {}

    # Generate with config or defaults
    generate_synthetic_data(
        n_patients=gen_config.get("n_patients", 1000),
        n_admissions_range=tuple(gen_config.get("n_admissions_per_patient", [1, 3])),
        icu_rate=gen_config.get("icu_admission_rate", 0.3),
        readmission_rate=gen_config.get("readmission_rate", 0.15),
        mortality_rate=gen_config.get("mortality_rate", 0.10),
        event_freq_hours=tuple(gen_config.get("event_frequency_hours", [1, 4])),
        random_state=gen_config.get("random_state", 42),
    )
