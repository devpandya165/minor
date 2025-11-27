import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import os  # For file handling


# ---------------------------
# 1. Holiday helper
# ---------------------------
def get_floating_holidays(year):
    """
    Returns approximate (Month, Day) tuples for major floating Indian holidays
    for a given year. Note: Exact dates vary based on regional calendars/moon phases.
    Using 2025 approximations based on current calendar.
    """
    # Using 2025 approximations as the current time is Oct 2025
    if year >= 2025:
        return [
            (3, 14),  # Holi (approx)
            (4, 1),   # Eid al-Fitr (approx)
            (8, 16),  # Janmashtami (approx)
            (10, 2),  # Dussehra (approx)
            (10, 20), # Diwali (approx)
            (11, 5)   # Guru Nanak Jayanti (approx)
        ]
    # Fallback/Example for 2024
    else:
        return [
            (3, 25),  # Holi (approx)
            (4, 11),  # Eid al-Fitr (approx)
            (8, 19),  # Janmashtami (approx)
            (10, 12), # Dussehra (approx)
            (11, 1),  # Diwali (approx)
            (11, 15)  # Guru Nanak Jayanti (approx)
        ]


# ---------------------------
# 2. Anomaly and missingness injectors
# ---------------------------
def inject_outliers(df, fraction=0.002, magnitude=0.4, random_state=42):
    """
    Randomly multiplies Gross_Load_MW by (1 ± magnitude)
    for a small fraction of rows to create spikes/drops.
    """
    if fraction <= 0:
        return df

    rng = np.random.default_rng(random_state)
    df = df.copy()
    n = len(df)
    k = max(1, int(n * fraction))

    indices = rng.choice(n, k, replace=False)
    signs = rng.choice([-1, 1], size=k)

    df.loc[indices, "Gross_Load_MW"] = (
        df.loc[indices, "Gross_Load_MW"] * (1 + signs * magnitude)
    )
    return df


def inject_missingness(df, frac_load=0.003, frac_weather=0.005, random_state=123):
    """
    Randomly sets some target and weather values to NaN to simulate missing data.
    """
    if frac_load <= 0 and frac_weather <= 0:
        return df

    rng = np.random.default_rng(random_state)
    df = df.copy()
    n = len(df)

    # Missing load
    if frac_load > 0:
        k_load = max(1, int(n * frac_load))
        idx_load = rng.choice(n, k_load, replace=False)
        df.loc[idx_load, "Gross_Load_MW"] = np.nan

    # Missing weather
    if frac_weather > 0:
        k_weather = max(1, int(n * frac_weather))
        idx_weather = rng.choice(n, k_weather, replace=False)
        weather_cols = ["Avg_Temp_C", "Avg_Humidity_Pct", "Temp_Change_6H"]
        df.loc[idx_weather, weather_cols] = np.nan

    return df


# ---------------------------
# 3. Main generator
# ---------------------------
def generate_indian_load_data_complete(
    start_date_str='2024-01-01',
    num_years=1,
    save_to_csv=True,
    save_realistic_version=True,
    add_anomalies=True,
    outlier_fraction=0.002,
    outlier_magnitude=0.4,
    missing_load_fraction=0.003,
    missing_weather_fraction=0.005,
):
    """
    Generate synthetic hourly load and weather data for all Indian states/UTs.

    - Keeps the ORIGINAL schema and columns:
      ['State_Code', 'Timestamp_UTC', 'Gross_Load_MW', 'Hour_Of_Day',
       'Day_Of_Week', 'Is_Weekend', 'Is_Holiday_State', 'Avg_Temp_C',
       'Temp_Change_6H', 'Avg_Humidity_Pct']
    - Adds more realistic patterns (trend, weekly effect, AR(1) noise).
    - Optionally creates a second CSV with anomalies and missingness.

    Parameters
    ----------
    start_date_str : str
        Start date in 'YYYY-MM-DD' format.
    num_years : int
        Number of years of hourly data to generate.
    save_to_csv : bool
        If True, saves the clean dataset to 'synthetic_indian_load_data.csv'.
    save_realistic_version : bool
        If True, saves a second file 'synthetic_indian_load_data_realistic.csv'
        with anomalies and missingness if add_anomalies=True.
    add_anomalies : bool
        If True, inject outliers and missing values in the realistic version.
    outlier_fraction : float
        Fraction of rows to treat as outliers in the realistic version.
    outlier_magnitude : float
        Relative spike/drop magnitude for outliers.
    missing_load_fraction : float
        Fraction of rows with missing Gross_Load_MW in realistic version.
    missing_weather_fraction : float
        Fraction of rows with missing weather fields in realistic version.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        (clean_df, realistic_df)
    """

    # --- 1. State/UT Parameters (same as your original) ---
    INDIAN_STATE_PARAMS = {
        'UP': {'Load_Multiplier': 30, 'Min_Temp': 5,  'Max_Temp': 42, 'Humidity_Factor': 0.5, 'Optimal_Temp': 20, 'Load_Type': 'U'},
        'MH': {'Load_Multiplier': 25, 'Min_Temp': 15, 'Max_Temp': 35, 'Humidity_Factor': 0.8, 'Optimal_Temp': 24, 'Load_Type': 'J'},
        'RJ': {'Load_Multiplier': 18, 'Min_Temp': 5,  'Max_Temp': 45, 'Humidity_Factor': 0.3, 'Optimal_Temp': 20, 'Load_Type': 'U'},
        'GJ': {'Load_Multiplier': 22, 'Min_Temp': 10, 'Max_Temp': 40, 'Humidity_Factor': 0.6, 'Optimal_Temp': 22, 'Load_Type': 'U'},
        'TN': {'Load_Multiplier': 20, 'Min_Temp': 20, 'Max_Temp': 38, 'Humidity_Factor': 0.9, 'Optimal_Temp': 26, 'Load_Type': 'J'},
        'KA': {'Load_Multiplier': 18, 'Min_Temp': 18, 'Max_Temp': 36, 'Humidity_Factor': 0.7, 'Optimal_Temp': 25, 'Load_Type': 'J'},
        'AP': {'Load_Multiplier': 15, 'Min_Temp': 22, 'Max_Temp': 40, 'Humidity_Factor': 0.7, 'Optimal_Temp': 28, 'Load_Type': 'J'},
        'TS': {'Load_Multiplier': 14, 'Min_Temp': 18, 'Max_Temp': 42, 'Humidity_Factor': 0.6, 'Optimal_Temp': 25, 'Load_Type': 'J'},
        'MP': {'Load_Multiplier': 13, 'Min_Temp': 8,  'Max_Temp': 40, 'Humidity_Factor': 0.5, 'Optimal_Temp': 22, 'Load_Type': 'U'},
        'WB': {'Load_Multiplier': 12, 'Min_Temp': 12, 'Max_Temp': 38, 'Humidity_Factor': 0.9, 'Optimal_Temp': 26, 'Load_Type': 'J'},
        'HR': {'Load_Multiplier': 10, 'Min_Temp': 5,  'Max_Temp': 40, 'Humidity_Factor': 0.5, 'Optimal_Temp': 20, 'Load_Type': 'U'},
        'PB': {'Load_Multiplier': 10, 'Min_Temp': 2,  'Max_Temp': 40, 'Humidity_Factor': 0.4, 'Optimal_Temp': 18, 'Load_Type': 'U'},
        'BH': {'Load_Multiplier': 9,  'Min_Temp': 8,  'Max_Temp': 38, 'Humidity_Factor': 0.7, 'Optimal_Temp': 24, 'Load_Type': 'J'},
        'OD': {'Load_Multiplier': 8,  'Min_Temp': 15, 'Max_Temp': 40, 'Humidity_Factor': 0.8, 'Optimal_Temp': 28, 'Load_Type': 'J'},
        'DL': {'Load_Multiplier': 8,  'Min_Temp': 3,  'Max_Temp': 43, 'Humidity_Factor': 0.5, 'Optimal_Temp': 20, 'Load_Type': 'U'},
        'KL': {'Load_Multiplier': 6,  'Min_Temp': 22, 'Max_Temp': 35, 'Humidity_Factor': 0.95, 'Optimal_Temp': 28, 'Load_Type': 'J'},
        'UK': {'Load_Multiplier': 4,  'Min_Temp': 0,  'Max_Temp': 30, 'Humidity_Factor': 0.6, 'Optimal_Temp': 18, 'Load_Type': 'INVERSE_U'},
        'CHH':{'Load_Multiplier': 5,  'Min_Temp': 10, 'Max_Temp': 40, 'Humidity_Factor': 0.5, 'Optimal_Temp': 22, 'Load_Type': 'U'},
        'AS': {'Load_Multiplier': 4,  'Min_Temp': 10, 'Max_Temp': 35, 'Humidity_Factor': 0.8, 'Optimal_Temp': 25, 'Load_Type': 'J'},
        'JH': {'Load_Multiplier': 4,  'Min_Temp': 10, 'Max_Temp': 38, 'Humidity_Factor': 0.6, 'Optimal_Temp': 22, 'Load_Type': 'U'},
        'HP': {'Load_Multiplier': 3,  'Min_Temp': -5, 'Max_Temp': 25, 'Humidity_Factor': 0.6, 'Optimal_Temp': 15, 'Load_Type': 'INVERSE_U'},
        'SK': {'Load_Multiplier': 0.5, 'Min_Temp': 0, 'Max_Temp': 20, 'Humidity_Factor': 0.6, 'Optimal_Temp': 12, 'Load_Type': 'INVERSE_U'},
        'GA': {'Load_Multiplier': 0.8, 'Min_Temp': 22, 'Max_Temp': 32, 'Humidity_Factor': 0.95, 'Optimal_Temp': 28, 'Load_Type': 'FLAT'},
        'AR': {'Load_Multiplier': 0.5, 'Min_Temp': 5, 'Max_Temp': 30, 'Humidity_Factor': 0.8, 'Optimal_Temp': 20, 'Load_Type': 'INVERSE_U'},
        'MN': {'Load_Multiplier': 0.6, 'Min_Temp': 10, 'Max_Temp': 30, 'Humidity_Factor': 0.7, 'Optimal_Temp': 20, 'Load_Type': 'J'},
        'ML': {'Load_Multiplier': 0.5, 'Min_Temp': 8, 'Max_Temp': 28, 'Humidity_Factor': 0.8, 'Optimal_Temp': 20, 'Load_Type': 'J'},
        'MZ': {'Load_Multiplier': 0.4, 'Min_Temp': 10, 'Max_Temp': 30, 'Humidity_Factor': 0.8, 'Optimal_Temp': 20, 'Load_Type': 'J'},
        'NL': {'Load_Multiplier': 0.5, 'Min_Temp': 10, 'Max_Temp': 28, 'Humidity_Factor': 0.7, 'Optimal_Temp': 20, 'Load_Type': 'J'},
        'TR': {'Load_Multiplier': 0.7, 'Min_Temp': 10, 'Max_Temp': 35, 'Humidity_Factor': 0.9, 'Optimal_Temp': 26, 'Load_Type': 'J'},
        'JK': {'Load_Multiplier': 2, 'Min_Temp': -5, 'Max_Temp': 35, 'Humidity_Factor': 0.4, 'Optimal_Temp': 15, 'Load_Type': 'INVERSE_U'},
        'LA': {'Load_Multiplier': 0.2, 'Min_Temp': -20,'Max_Temp': 20, 'Humidity_Factor': 0.2, 'Optimal_Temp': 10, 'Load_Type': 'INVERSE_U'},
        'PD': {'Load_Multiplier': 0.3, 'Min_Temp': 22, 'Max_Temp': 35, 'Humidity_Factor': 0.9, 'Optimal_Temp': 28, 'Load_Type': 'J'},
        'CH': {'Load_Multiplier': 0.5, 'Min_Temp': 5, 'Max_Temp': 40, 'Humidity_Factor': 0.5, 'Optimal_Temp': 20, 'Load_Type': 'U'},
        'DNHD': {'Load_Multiplier': 1, 'Min_Temp': 15, 'Max_Temp': 35, 'Humidity_Factor': 0.8, 'Optimal_Temp': 24, 'Load_Type': 'J'},
        'AN': {'Load_Multiplier': 0.1, 'Min_Temp': 23, 'Max_Temp': 32, 'Humidity_Factor': 0.9, 'Optimal_Temp': 28, 'Load_Type': 'FLAT'},
        'LD': {'Load_Multiplier': 0.05, 'Min_Temp': 25, 'Max_Temp': 32, 'Humidity_Factor': 0.9, 'Optimal_Temp': 28, 'Load_Type': 'FLAT'},
    }

    FIXED_HOLIDAYS = [(1, 26), (8, 15), (10, 2), (1, 1)]  # includes New Year

    # --- 2. Create Base Time Series for All States/UTs (hourly) ---
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date + relativedelta(years=num_years) - relativedelta(hours=1)

    time_series = pd.date_range(start=start_date, end=end_date, freq='H')
    state_codes = list(INDIAN_STATE_PARAMS.keys())

    df = pd.DataFrame({
        'Timestamp_UTC': np.tile(time_series, len(state_codes)),
        'State_Code': np.repeat(state_codes, len(time_series))
    }).sort_values(by=['State_Code', 'Timestamp_UTC']).reset_index(drop=True)

    # --- 3. Calendar Features & Dynamic Holidays (same as original) ---
    df['Hour_Of_Day'] = df['Timestamp_UTC'].dt.hour
    df['Day_Of_Week'] = df['Timestamp_UTC'].dt.dayofweek + 1
    df['Is_Weekend'] = df['Day_Of_Week'].isin([6, 7]).astype(int)

    holiday_dates = set(FIXED_HOLIDAYS)
    for year in range(start_date.year, end_date.year + 1):
        holiday_dates.update(get_floating_holidays(year))

    def check_holiday(ts):
        return (ts.month, ts.day) in holiday_dates

    df['Is_Holiday_State'] = df['Timestamp_UTC'].apply(check_holiday).astype(int)

    # Initialize columns
    df[['Avg_Temp_C', 'Avg_Humidity_Pct', 'Temp_Change_6H', 'Gross_Load_MW']] = np.nan

    # --- 4. Generate Data per State/UT with extra realism ---
    for state, params in INDIAN_STATE_PARAMS.items():
        state_filter = (df['State_Code'] == state)
        sub_df = df[state_filter].copy()

        # Slight yearly randomization of base load and optimal temp
        yearly_factor = 1 + np.random.normal(0, 0.02)
        base_load_mw = params['Load_Multiplier'] * yearly_factor * 1000
        optimal_temp = params['Optimal_Temp'] + np.random.normal(0, 0.5)

        # Weather Generation
        day_of_year = sub_df['Timestamp_UTC'].dt.dayofyear
        base_temp = (params['Min_Temp'] + params['Max_Temp']) / 2
        amplitude = (params['Max_Temp'] - params['Min_Temp']) / 2

        seasonal_cycle = amplitude * np.sin(2 * np.pi * (day_of_year - 110) / 365)
        diurnal_cycle = 3 * np.sin(2 * np.pi * (sub_df['Hour_Of_Day'] - 15) / 24)

        noise_temp = np.random.normal(0, 1.5, len(sub_df))
        sub_df['Avg_Temp_C'] = base_temp + seasonal_cycle + diurnal_cycle + noise_temp

        sub_df['Avg_Humidity_Pct'] = (
            (80 * params['Humidity_Factor']) -
            (sub_df['Avg_Temp_C'] * 0.5) +
            np.random.normal(0, 5, len(sub_df))
        )
        sub_df['Avg_Humidity_Pct'] = np.clip(sub_df['Avg_Humidity_Pct'], 20, 100)
        sub_df['Temp_Change_6H'] = sub_df['Avg_Temp_C'].diff(periods=6).fillna(0)

        # Load Generation: temp effect
        temp_diff = sub_df['Avg_Temp_C'] - optimal_temp

        if params['Load_Type'] == 'U':
            temp_effect = 25 * temp_diff**2
        elif params['Load_Type'] == 'J':
            temp_effect = 30 * np.maximum(0, temp_diff)**2
        elif params['Load_Type'] == 'INVERSE_U':
            temp_effect = 50 * np.minimum(0, temp_diff)**2 + np.maximum(0, temp_diff * 50)
        else:
            temp_effect = 0

        # Hourly peak (evening)
        hour_peak_factor = 0.1 * base_load_mw * np.cos(
            2 * np.pi * (sub_df['Hour_Of_Day'] - 19) / 24
        )

        # Weekend/holiday reduction
        day_off_reduction = 0.18 * base_load_mw * (
            sub_df['Is_Weekend'] + sub_df['Is_Holiday_State']
        )

        # Time index in hours for trend
        time_index_hours = (
            (sub_df['Timestamp_UTC'] - sub_df['Timestamp_UTC'].min())
            .dt.total_seconds() / 3600.0
        )
        trend_slope = np.random.normal(0.0005, 0.0003)
        trend_component = trend_slope * time_index_hours * base_load_mw

        # Weekly pattern (1=Mon..7=Sun)
        dow = sub_df['Day_Of_Week'].values
        weekly_multipliers = np.array([0.02, 0.01, 0.00, -0.01, 0.03, -0.05, -0.03])
        weekly_pattern = weekly_multipliers[dow - 1] * base_load_mw

        # AR(1) noise
        rho = 0.8
        eps = np.random.normal(0, 1, len(sub_df))
        ar_noise = np.zeros(len(sub_df))
        for i in range(1, len(sub_df)):
            ar_noise[i] = rho * ar_noise[i - 1] + eps[i]

        noise_level = base_load_mw / 10000
        ar_noise = ar_noise * (noise_level * 500)

        # Final load
        sub_df['Gross_Load_MW'] = (
            base_load_mw +
            temp_effect +
            hour_peak_factor -
            day_off_reduction +
            trend_component +
            weekly_pattern +
            ar_noise
        )

        sub_df['Gross_Load_MW'] = np.clip(sub_df['Gross_Load_MW'], base_load_mw * 0.5, None)

        df.loc[state_filter, ['Avg_Temp_C', 'Avg_Humidity_Pct', 'Temp_Change_6H', 'Gross_Load_MW']] = \
            sub_df[['Avg_Temp_C', 'Avg_Humidity_Pct', 'Temp_Change_6H', 'Gross_Load_MW']]

    # --- 5. Final cleanup (same columns as original) ---
    clean_df = df[[
        'State_Code',
        'Timestamp_UTC',
        'Gross_Load_MW',
        'Hour_Of_Day',
        'Day_Of_Week',
        'Is_Weekend',
        'Is_Holiday_State',
        'Avg_Temp_C',
        'Temp_Change_6H',
        'Avg_Humidity_Pct'
    ]].copy()

    # Create realistic version (with anomalies/missingness) if requested
    if add_anomalies:
        realistic_df = inject_outliers(
            clean_df,
            fraction=outlier_fraction,
            magnitude=outlier_magnitude,
            random_state=42
        )
        realistic_df = inject_missingness(
            realistic_df,
            frac_load=missing_load_fraction,
            frac_weather=missing_weather_fraction,
            random_state=123
        )
    else:
        realistic_df = clean_df.copy()

    # Save files
    if save_to_csv:
        base_path = 'synthetic_indian_load_data.csv'
        clean_df.to_csv(base_path, index=False)
        print("\n----------------------------------------------------------------------------------")
        print(f"✅ CLEAN data successfully generated and stored to: {os.path.abspath(base_path)}")
        print(f"Total Rows: {len(clean_df)} (1 year hourly data for all 36 entities)")
        print("----------------------------------------------------------------------------------")

    if save_realistic_version:
        realistic_path = 'synthetic_indian_load_data_realistic.csv'
        realistic_df.to_csv(realistic_path, index=False)
        print(f"✅ REALISTIC data (with anomalies) stored to: {os.path.abspath(realistic_path)}")

    return clean_df, realistic_df


# --- EXECUTION ---
if __name__ == "__main__":
    clean_df, realistic_df = generate_indian_load_data_complete(
        start_date_str='2024-01-01',
        num_years=1,
        save_to_csv=True,
        save_realistic_version=True,
        add_anomalies=True,
        outlier_fraction=0.002,
        outlier_magnitude=0.4,
        missing_load_fraction=0.003,
        missing_weather_fraction=0.005,
    )

    print("\n--- Sample of CLEAN data (head) ---")
    print(clean_df.head())
    print("\n--- Sample of REALISTIC data (head) ---")
    print(realistic_df.head())