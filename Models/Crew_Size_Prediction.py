import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
predictions_path = os.path.join(
    project_root,
    "Data_Sources",
    "Data_Cleaned",
    "Predictions",
    "Segmented_Visitor_Demand_Prediction.csv",
)
modelling_path = os.path.join(
    project_root, "Data_Sources", "Data_Cleaned", "Modelling", "Table_for_modelling.csv"
)

try:
    predictions_df = pd.read_csv(predictions_path)
    df = pd.read_csv(modelling_path)
except FileNotFoundError:
    predictions_df = None  # or create an empty DataFrame
    df = None
    print(f"Warning: Could not load {predictions_path} or {modelling_path}")


cols_to_merge = [
    "Recreatief NL_pred",
    "Recreatief NL_actual",
    "Recreatief Buitenland_pred",
    "Recreatief Buitenland_actual",
    "PO_pred",
    "PO_actual",
    "VO_pred",
    "VO_actual",
    "Student_pred",
    "Student_actual",
    "Extern_pred",
    "Extern_actual",
    "Total Visitors_pred",
    "Total Visitors_actual",
    "Date",
]

merged_df = df.merge(predictions_df[cols_to_merge], on="Date", how="left")
new_df = merged_df.iloc[30:, :]


# Model Crew Size Prediction
class CrewSizePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_mapping = {}

        # Define crew size categories in order of capacity
        self.crew_size_order = [
            "Gesloten",
            "Gesloten maandag",
            "A min",
            "A",
            "B",
            "C",
            "D",
        ]

    def clean_crew_size_data(self, df):
        """Clean and prepare crew size data"""
        df_clean = df.copy()

        # Handle NaN values in maat_visitors
        # You might want to fill these based on business logic
        df_clean["maat_visitors"] = df_clean["maat_visitors"].fillna("Unknown")

        # Create a mapping for crew sizes to numerical values for ordering
        crew_mapping = {size: idx for idx, size in enumerate(self.crew_size_order)}
        crew_mapping["Unknown"] = -1  # For NaN values

        df_clean["crew_size_numeric"] = df_clean["maat_visitors"].map(crew_mapping)

        print("Crew size distribution:")
        print(df_clean["maat_visitors"].value_counts())

        return df_clean

    def engineer_crew_features(self, df, historical_df=None):
        """Engineer features for crew size prediction, with optional historical injection"""
        df_features = df.copy()

        print("=== FEATURE ENGINEERING DEBUG ===")
        print(f"Initial DataFrame shape: {df_features.shape}")

        # Basic time-based features
        if "Date" in df.columns:
            df_features["Date"] = pd.to_datetime(df_features["Date"])
            df_features = df_features.sort_values("Date").reset_index(drop=True)
            df_features["day_of_week"] = df_features["Date"].dt.dayofweek
            df_features["is_weekend"] = (
                df_features["day_of_week"].isin([5, 6]).astype(int)
            )
            df_features["is_monday"] = (df_features["day_of_week"] == 0).astype(int)
            df_features["month"] = df_features["Date"].dt.month
            df_features["is_summer"] = df_features["month"].isin([6, 7, 8]).astype(int)

        # Weather
        if "MeanTemp_C" in df.columns and "Precipitation_mm" in df.columns:
            df_features["good_weather"] = (
                (df_features["MeanTemp_C"] > 15) & (df_features["Precipitation_mm"] < 1)
            ).astype(int)

            df_features["bad_weather"] = (
                (df_features["MeanTemp_C"] < 10) | (df_features["Precipitation_mm"] > 5)
            ).astype(int)

        # Holidays
        holiday_cols = [col for col in df.columns if "holiday" in col.lower()]
        if holiday_cols:
            df_features["any_holiday"] = df_features[holiday_cols].max(axis=1)

        # Crew size encoding
        if "maat_visitors" in df.columns:
            crew_size_mapping = {
                "Gesloten": 0,
                "Gesloten maandag": 1,
                "A min": 2,
                "A": 3,
                "B": 4,
                "C": 5,
                "D": 6,
            }
            df_features["crew_size_numeric"] = df_features["maat_visitors"].map(
                crew_size_mapping
            )
            df_features["crew_size_numeric"] = df_features["crew_size_numeric"].fillna(
                3
            )

        # SEASONAL
        if "Date" in df.columns:
            df_features["weekday"] = df_features["Date"].dt.dayofweek
            df_features["weekday_crew_avg"] = (
                df_features.groupby("weekday")["crew_size_numeric"]
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )
            df_features["weekday_crew_avg"] = df_features["weekday_crew_avg"].shift(1)

        # INTERACTION
        if (
            "high_capacity_day" in df_features.columns
            and "good_weather" in df_features.columns
        ):
            df_features["high_visitors_good_weather"] = (
                df_features["high_capacity_day"] * df_features["good_weather"]
            )

        if "is_weekend" in df_features.columns and "any_holiday" in df_features.columns:
            df_features["weekend_holiday"] = (
                df_features["is_weekend"] * df_features["any_holiday"]
            )

        # === LAGGED & ROLLING FEATURES ===
        print("\n=== LAGGED & ROLLING FEATURES ===")
        if historical_df is None:
            # Simple case: normal training → use shift() and rolling()
            lags = [1, 7, 14]

            for lag in lags:
                df_features[f"crew_size_lag_{lag}"] = df_features[
                    "crew_size_numeric"
                ].shift(lag)
            df_features["crew_size_last_week"] = df_features["crew_size_numeric"].shift(
                7
            )

            def numeric_mode(x):
                mode_result = x.mode()
                return mode_result.iloc[0] if len(mode_result) > 0 else 3

            def crew_stability(x):
                if x.mean() > 0:
                    return (x == x.mode().iloc[0]).mean()
                else:
                    return 0.6

            for window in [7, 14]:
                df_features[f"crew_mode_numeric_{window}"] = (
                    df_features["crew_size_numeric"]
                    .shift(1)
                    .rolling(window=window)
                    .apply(numeric_mode, raw=False)
                )
                df_features[f"crew_stability_{window}"] = (
                    df_features["crew_size_numeric"]
                    .shift(1)
                    .rolling(window=window)
                    .apply(crew_stability, raw=False)
                )

                df_features[f"good_weather_freq_{window}"] = (
                    df_features["good_weather"].shift(1).rolling(window=window).mean()
                )

                df_features[f"holiday_density_{window}"] = (
                    df_features["any_holiday"].shift(1).rolling(window=window).mean()
                )

        else:
            # Historical fallback mode (prediction time)
            print("Injecting lagged/rolling features using historical data...")
            for index, row in df_features.iterrows():
                prediction_date = row["Date"]
                df_features.loc[index] = self.get_historical_lagged_features(
                    df_features.loc[[index]], historical_df, prediction_date
                ).iloc[0]

        # Final NA cleanup
        df_features.fillna(method="ffill", inplace=True)
        df_features.fillna(method="bfill", inplace=True)
        df_features.fillna(0, inplace=True)

        print("=== END FEATURE ENGINEERING DEBUG ===\n")
        return df_features

    # def engineer_crew_features(self, df, historical_data=None):
    #     """Engineer features specifically for crew size prediction with rolling and lagged features"""
    #     df_features = df.copy()
    #     print("=== FEATURE ENGINEERING DEBUG ===")
    #     print("Columns before engineering")
    #     print(df_features.columns)
    #     print(f"Initial DataFrame shape: {df_features.shape}")
    #     print(f"Initial NaN values: {df_features.isna().sum().sum()}")

    #     # Check initial NaN values by column
    #     initial_nan_cols = df_features.columns[df_features.isna().any()].tolist()
    #     if initial_nan_cols:
    #         print("\nInitial columns with NaN values:")
    #         for col in initial_nan_cols:
    #             nan_count = df_features[col].isna().sum()
    #             print(f"  {col}: {nan_count} NaN values ({nan_count/len(df_features)*100:.1f}%)")

    #     # Ensure Date is datetime
    #     if 'Date' in df.columns:
    #         df_features['Date'] = pd.to_datetime(df_features['Date'])
    #         df_features = df_features.sort_values('Date').reset_index(drop=True)

    #     # === VISITOR RATIOS ===
    #     print("\n--- Creating visitor ratios ---")
    #     if 'Total Visitors_pred' in df.columns:
    #         total_pred = df_features['Total Visitors_pred']
    #         print(f"Total Visitors_pred NaN count: {total_pred.isna().sum()}")

    #         df_features['recreatief_nl_ratio'] = df_features['Recreatief NL_pred'] / (total_pred + 1)
    #         df_features['recreatief_buitenland_ratio'] = df_features['Recreatief Buitenland_pred'] / (total_pred + 1)
    #         df_features['educational_ratio'] = (
    #             df_features['PO_pred'] + df_features['VO_pred'] + df_features['Student_pred']
    #         ) / (total_pred + 1)
    #         df_features['extern_ratio'] = df_features['Extern_pred'] / (total_pred + 1)

    #         # Check for NaN in ratios
    #         ratio_cols = ['recreatief_nl_ratio', 'recreatief_buitenland_ratio', 'educational_ratio', 'extern_ratio']
    #         for col in ratio_cols:
    #             nan_count = df_features[col].isna().sum()
    #             if nan_count > 0:
    #                 print(f"  WARNING: {col} has {nan_count} NaN values")

    #     # === TIME-BASED FEATURES ===
    #     print("\n--- Creating time-based features ---")
    #     if 'Date' in df.columns:
    #         df_features['day_of_week'] = df_features['Date'].dt.dayofweek
    #         df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
    #         df_features['is_monday'] = (df_features['day_of_week'] == 0).astype(int)
    #         df_features['month'] = df_features['Date'].dt.month
    #         df_features['is_summer'] = df_features['month'].isin([6, 7, 8]).astype(int)

    #     # === WEATHER FEATURES ===
    #     print("\n--- Creating weather features ---")
    #     if 'MeanTemp_C' in df.columns and 'Precipitation_mm' in df.columns:
    #         temp_nan = df_features['MeanTemp_C'].isna().sum()
    #         precip_nan = df_features['Precipitation_mm'].isna().sum()
    #         print(f"MeanTemp_C NaN count: {temp_nan}")
    #         print(f"Precipitation_mm NaN count: {precip_nan}")

    #         df_features['good_weather'] = (
    #             (df_features['MeanTemp_C'] > 15) &
    #             (df_features['Precipitation_mm'] < 1)
    #         ).astype(int)
    #         df_features['bad_weather'] = (
    #             (df_features['MeanTemp_C'] < 10) |
    #             (df_features['Precipitation_mm'] > 5)
    #         ).astype(int)

    #         # Check for NaN in weather features
    #         weather_nan_good = df_features['good_weather'].isna().sum()
    #         weather_nan_bad = df_features['bad_weather'].isna().sum()
    #         if weather_nan_good > 0 or weather_nan_bad > 0:
    #             print(f"  WARNING: good_weather has {weather_nan_good} NaN, bad_weather has {weather_nan_bad} NaN")

    #     # === HOLIDAY FEATURES ===
    #     print("\n--- Creating holiday features ---")
    #     holiday_cols = [col for col in df.columns if 'holiday' in col.lower()]
    #     if holiday_cols:
    #         print(f"Holiday columns found: {holiday_cols}")
    #         for col in holiday_cols:
    #             nan_count = df_features[col].isna().sum()
    #             if nan_count > 0:
    #                 print(f"  {col} has {nan_count} NaN values")

    #         df_features['any_holiday'] = df_features[holiday_cols].max(axis=1)
    #         any_holiday_nan = df_features['any_holiday'].isna().sum()
    #         if any_holiday_nan > 0:
    #             print(f"  WARNING: any_holiday has {any_holiday_nan} NaN values")

    #     # === OPERATIONAL FEATURES ===
    #     print("\n--- Creating operational features ---")
    #     if 'is_open' in df.columns:
    #         is_open_nan = df_features['is_open'].isna().sum()
    #         print(f"is_open NaN count before fillna: {is_open_nan}")
    #         df_features['is_open'] = df_features['is_open'].fillna(1)
    #         print(f"is_open NaN count after fillna: {df_features['is_open'].isna().sum()}")

    #     # === CAPACITY FEATURES ===
    #     print("\n--- Creating capacity features ---")
    #     if 'Total Visitors_pred' in df.columns:
    #         total_visitors_nan = df_features['Total Visitors_pred'].isna().sum()
    #         print(f"Total Visitors_pred NaN count: {total_visitors_nan}")

    #         if total_visitors_nan == 0:  # Only create if no NaN
    #             q80 = df_features['Total Visitors_pred'].quantile(0.8)
    #             q20 = df_features['Total Visitors_pred'].quantile(0.2)
    #             df_features['high_capacity_day'] = (df_features['Total Visitors_pred'] > q80).astype(int)
    #             df_features['low_capacity_day'] = (df_features['Total Visitors_pred'] < q20).astype(int)
    #         else:
    #             print("  WARNING: Skipping capacity features due to NaN in Total Visitors_pred")
    #             df_features['high_capacity_day'] = 0
    #             df_features['low_capacity_day'] = 0

    #     # === LAGGED FEATURES ===
    #     print("\n--- Creating lagged features ---")
    #     visitor_pred_cols = [col for col in df_features.columns if col.endswith('_pred')]
    #     print(f"Visitor prediction columns: {visitor_pred_cols}")

    #     lags = [1, 7, 14]
    #     for lag in lags:
    #         print(f"  Creating lag-{lag} features...")
    #         for col in visitor_pred_cols:
    #             if col in df_features.columns:
    #                 original_nan = df_features[col].isna().sum()
    #                 df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
    #                 lagged_nan = df_features[f'{col}_lag_{lag}'].isna().sum()
    #                 if original_nan > 0 or lagged_nan != lag:  # Expected NaN = lag value
    #                     print(f"    {col}_lag_{lag}: original NaN={original_nan}, lagged NaN={lagged_nan}")

    #         # Total visitors lag
    #         if 'Total Visitors_pred' in df.columns:
    #             df_features[f'total_visitors_lag_{lag}'] = df_features['Total Visitors_pred'].shift(lag)

    #     # === CREW SIZE ENCODING ===
    #     print("\n--- Creating crew size features ---")
    #     if 'maat_visitors' in df.columns:
    #         crew_size_mapping = {
    #             'Gesloten': 0, 'Gesloten maandag': 1, 'A min': 2,
    #             'A': 3, 'B': 4, 'C': 5, 'D': 6
    #         }

    #         crew_nan_before = df_features['maat_visitors'].isna().sum()
    #         print(f"maat_visitors NaN count before mapping: {crew_nan_before}")

    #         df_features['crew_size_numeric'] = df_features['maat_visitors'].map(crew_size_mapping)
    #         df_features['crew_size_numeric'] = df_features['crew_size_numeric'].fillna(-1)

    #         crew_nan_after = df_features['crew_size_numeric'].isna().sum()
    #         print(f"crew_size_numeric NaN count after mapping: {crew_nan_after}")

    #         # Lagged crew sizes
    #         for lag in lags:
    #             df_features[f'crew_size_lag_{lag}'] = df_features['crew_size_numeric'].shift(lag)

    #         df_features['crew_size_last_week'] = df_features['crew_size_numeric'].shift(7)

    #     # === ROLLING FEATURES ===
    #     print("\n--- Creating rolling features ---")
    #     windows = [7, 14]

    #     for window in windows:
    #         print(f"  Creating {window}-day rolling features...")

    #         # Rolling visitor statistics
    #         for col in visitor_pred_cols:
    #             if col in df_features.columns:
    #                 base_name = col.replace('_pred', '')
    #                 try:
    #                     df_features[f'{col}_rolling_mean_{window}'] = df_features[col].shift(1).rolling(window=window).mean()
    #                     df_features[f'{col}_rolling_std_{window}'] = df_features[col].shift(1).rolling(window=window).std()
    #                     df_features[f'{col}_rolling_max_{window}'] = df_features[col].shift(1).rolling(window=window).max()

    #                     # Check for excessive NaN (should only be first window-1 + 1 values)
    #                     expected_nan = window
    #                     actual_nan = df_features[f'{col}_rolling_mean_{window}'].isna().sum()
    #                     if actual_nan > expected_nan:
    #                         print(f"    WARNING: {col}_rolling_mean_{window} has {actual_nan} NaN (expected ~{expected_nan})")

    #                 except Exception as e:
    #                     print(f"    ERROR creating rolling features for {col}: {e}")

    #         # Rolling crew size patterns
    #         if 'crew_size_numeric' in df_features.columns:
    #             try:
    #                 def numeric_mode(x):
    #                     if len(x) == 0 or x.isna().all():
    #                         return -1
    #                     mode_result = x.mode()
    #                     return mode_result.iloc[0] if len(mode_result) > 0 else -1

    #                 def crew_stability(x):
    #                     if len(x) == 0 or x.isna().all():
    #                         return 0
    #                     mode_result = x.mode()
    #                     if len(mode_result) == 0:
    #                         return 0
    #                     most_common = mode_result.iloc[0]
    #                     return (x == most_common).mean()

    #                 df_features[f'crew_mode_numeric_{window}'] = (
    #                     df_features['crew_size_numeric'].shift(1).rolling(window=window).apply(numeric_mode, raw=False)
    #                 )

    #                 df_features[f'crew_stability_{window}'] = (
    #                     df_features['crew_size_numeric'].shift(1).rolling(window=window).apply(crew_stability, raw=False)
    #                 )

    #             except Exception as e:
    #                 print(f"    ERROR creating crew rolling features for window {window}: {e}")

    #         # Rolling weather patterns
    #         if 'good_weather' in df_features.columns:
    #             df_features[f'good_weather_freq_{window}'] = (
    #                 df_features['good_weather'].shift(1).rolling(window=window).mean()
    #             )

    #         # Rolling holiday density
    #         if 'any_holiday' in df_features.columns:
    #             df_features[f'holiday_density_{window}'] = (
    #                 df_features['any_holiday'].shift(1).rolling(window=window).mean()
    #             )

    #     # === SEASONAL PATTERNS ===
    #     print("\n--- Creating seasonal patterns ---")
    #     if 'Date' in df.columns:
    #         df_features['weekday'] = df_features['Date'].dt.dayofweek

    #         if 'crew_size_numeric' in df_features.columns:
    #             try:
    #                 weekday_crew_avg = df_features.groupby('weekday')['crew_size_numeric'].expanding().mean()
    #                 df_features['weekday_crew_avg'] = weekday_crew_avg.reset_index(level=0, drop=True)
    #                 df_features['weekday_crew_avg'] = df_features['weekday_crew_avg'].shift(1)

    #                 weekday_avg_nan = df_features['weekday_crew_avg'].isna().sum()
    #                 print(f"  weekday_crew_avg NaN count: {weekday_avg_nan}")

    #             except Exception as e:
    #                 print(f"  ERROR creating weekday patterns: {e}")

    #     # === INTERACTION FEATURES ===
    #     print("\n--- Creating interaction features ---")
    #     if 'Total Visitors_pred' in df_features.columns and 'good_weather' in df_features.columns:
    #         if 'high_capacity_day' in df_features.columns:
    #             df_features['high_visitors_good_weather'] = (
    #                 df_features['high_capacity_day'] * df_features['good_weather']
    #             )

    #     if 'is_weekend' in df_features.columns and 'any_holiday' in df_features.columns:
    #         df_features['weekend_holiday'] = (
    #             df_features['is_weekend'] * df_features['any_holiday']
    #         )

    #     # === TREND FEATURES ===
    #     print("\n--- Creating trend features ---")
    #     if 'Total Visitors_pred' in df_features.columns:
    #         df_features['visitor_trend_3d'] = (
    #             df_features['Total Visitors_pred'] - df_features['Total Visitors_pred'].shift(3)
    #         )
    #         df_features['visitor_trend_7d'] = (
    #             df_features['Total Visitors_pred'] - df_features['Total Visitors_pred'].shift(7)
    #         )

    #     # === FINAL NaN ANALYSIS ===
    #     print("\n=== FINAL NaN ANALYSIS ===")
    #     print(f"Final DataFrame shape: {df_features.shape}")
    #     total_nan = df_features.isna().sum().sum()
    #     print(f"Total NaN values: {total_nan}")

    #     if total_nan > 0:
    #         print("\nColumns with NaN values (final):")
    #         nan_summary = df_features.isna().sum()
    #         nan_cols = nan_summary[nan_summary > 0].sort_values(ascending=False)

    #         for col, count in nan_cols.items():
    #             percentage = (count / len(df_features)) * 100
    #             print(f"  {col}: {count} NaN values ({percentage:.1f}%)")

    #             # Show some sample values for debugging
    #             non_nan_values = df_features[col].dropna()
    #             if len(non_nan_values) > 0:
    #                 print(f"    Sample non-NaN values: {non_nan_values.head(3).tolist()}")
    #             else:
    #                 print(f"    ALL VALUES ARE NaN!")

    #     print("=== END FEATURE ENGINEERING DEBUG ===\n")

    #     return df_features

    def select_features(self, df):
        """Select relevant features for crew size prediction including lagged and rolling features"""
        # Primary features: predicted visitor numbers
        feature_cols = [col for col in df.columns if col.endswith("_pred")]

        # Basic engineered features
        engineered_features = [
            "recreatief_nl_ratio",
            "recreatief_buitenland_ratio",
            "educational_ratio",
            "extern_ratio",
            "day_of_week",
            "is_weekend",
            "is_monday",
            "month",
            "is_summer",
            "good_weather",
            "bad_weather",
            "any_holiday",
            "is_open",
            "high_capacity_day",
            "low_capacity_day",
        ]

        # Weather and operational features
        additional_features = [
            "MeanTemp_C",
            "Precipitation_mm",
            "school_holiday",
            "public_holiday",
            "Events_in_Ams",
            "hotel_occupancy_index",
            "peak_season_flag",
        ]

        # === LAGGED FEATURES ===
        lagged_features = []

        # Visitor prediction lags (1, 7, 14 days)
        visitor_pred_base = [col for col in df.columns if col.endswith("_pred")]
        for base_col in visitor_pred_base:
            for lag in [1, 7, 14]:
                lagged_features.append(f"{base_col}_lag_{lag}")

        # Total visitors lags
        for lag in [1, 7, 14]:
            lagged_features.append(f"total_visitors_lag_{lag}")

        # Historical crew size lags (if available)
        crew_lag_features = [
            "crew_size_lag_1",
            "crew_size_lag_7",
            "crew_size_lag_14",
            "crew_size_last_week",
        ]
        lagged_features.extend(crew_lag_features)

        # === ROLLING FEATURES ===
        rolling_features = []

        # Rolling visitor statistics (7, 14, 30 day windows)
        for window in [7, 14, 30]:
            # For each visitor prediction column
            for base_col in visitor_pred_base:
                rolling_features.extend(
                    [
                        f"{base_col}_rolling_mean_{window}",
                        f"{base_col}_rolling_std_{window}",
                        f"{base_col}_rolling_max_{window}",
                    ]
                )

        # Rolling crew size patterns
        crew_rolling_features = []
        for window in [7, 14, 30]:
            crew_rolling_features.extend(
                [
                    f"crew_mode_numeric_{window}",  # Most frequent crew size
                    f"crew_stability_{window}",  # How consistent crew sizing was
                ]
            )
        rolling_features.extend(crew_rolling_features)

        # Rolling weather and holiday patterns
        weather_rolling_features = []
        for window in [7, 14, 30]:
            weather_rolling_features.extend(
                [
                    f"good_weather_freq_{window}",  # Good weather frequency
                    f"holiday_density_{window}",  # Holiday density
                ]
            )
        rolling_features.extend(weather_rolling_features)

        # === SEASONAL PATTERN FEATURES ===
        seasonal_features = [
            "weekday_crew_avg",  # Average crew size for this weekday
        ]

        # === INTERACTION FEATURES ===
        interaction_features = [
            "high_visitors_good_weather",  # High capacity + good weather
            "weekend_holiday",  # Weekend + holiday interaction
        ]

        # === TREND FEATURES ===
        trend_features = [
            "visitor_trend_3d",  # 3-day visitor change
            "visitor_trend_7d",  # 7-day visitor change
        ]

        # === COMBINE ALL FEATURE CATEGORIES ===
        all_features = (
            feature_cols
            + engineered_features  # Visitor predictions
            + additional_features  # Basic engineered features
            + lagged_features  # Weather/operational features
            + rolling_features  # Lagged features
            + seasonal_features  # Rolling window features
            + interaction_features  # Seasonal patterns
            + trend_features  # Feature interactions  # Trend features
        )

        # Filter to only include features that actually exist in the dataframe
        selected_features = [col for col in all_features if col in df.columns]

        # Print feature summary
        feature_categories = {
            "Visitor Predictions": [col for col in feature_cols if col in df.columns],
            "Basic Features": [col for col in engineered_features if col in df.columns],
            "Weather/Operational": [
                col for col in additional_features if col in df.columns
            ],
            "Lagged Features": [col for col in lagged_features if col in df.columns],
            "Rolling Features": [col for col in rolling_features if col in df.columns],
            "Seasonal Features": [
                col for col in seasonal_features if col in df.columns
            ],
            "Interaction Features": [
                col for col in interaction_features if col in df.columns
            ],
            "Trend Features": [col for col in trend_features if col in df.columns],
        }

        print("=== FEATURE SELECTION SUMMARY ===")
        for category, features in feature_categories.items():
            print(f"{category}: {len(features)} features")
            if len(features) <= 5:  # Show all if 5 or fewer
                print(f"  {features}")
            else:  # Show first 3 and last 2 if more than 5
                print(f"  {features[:3]} ... {features[-2:]}")

        print(f"\nTotal selected features: {len(selected_features)}")

        return selected_features

    def prepare_data(self, df):
        """Prepare data for training"""
        # Clean crew size data
        df_clean = self.clean_crew_size_data(df)

        # Remove rows where museum is closed or crew size is unknown
        # You might want to predict these separately
        df_model = df_clean[
            (
                ~df_clean["maat_visitors"].isin(
                    ["Gesloten", "Unknown", "Gesloten maandag"]
                )
            )
            & (df_clean["is_open"] == 1)
        ].copy()

        print(f"Training data shape after filtering: {df_model.shape}")
        print("Remaining crew size distribution:")
        print(df_model["maat_visitors"].value_counts())

        # Engineer features
        df_features = self.engineer_crew_features(df_model)

        # Select features
        feature_cols = self.select_features(df_features)

        print("check if hotel_occupancy_index and peak_season_flag are in df_features:")
        print("hotel_occupancy_index:", "hotel_occupancy_index" in df_features.columns)
        print("peak_season_flag:", "peak_season_flag" in df_features.columns)

        X = df_features[feature_cols]
        y = df_features["maat_visitors"]

        # Handle any remaining NaN values
        X = X.fillna(X.median())

        return X, y, feature_cols

    def get_historical_lagged_features(self, input_df, historical_df, prediction_date):
        """Get lagged and rolling features from historical data for crew size prediction"""

        hist_df = historical_df.copy()
        hist_df["Date"] = pd.to_datetime(hist_df["Date"])

        # Calculate gap
        last_historical_date = hist_df["Date"].max()
        days_gap = (prediction_date - last_historical_date).days

        pred_weekday = prediction_date.weekday()
        pred_month = prediction_date.month
        weekday_name = prediction_date.strftime("%A")

        # Prepare lagged features dict
        lagged_features = {}

        # Crew size fallback mapping
        crew_mapping = {
            "Gesloten": 0,
            "Gesloten maandag": 1,
            "A min": 2,
            "A": 3,
            "B": 4,
            "C": 5,
            "D": 6,
        }
        if (
            "crew_size_numeric" not in hist_df.columns
            and "maat_visitors" in hist_df.columns
        ):
            hist_df["crew_size_numeric"] = (
                hist_df["maat_visitors"].map(crew_mapping).fillna(3)
            )

        # Add good_weather if missing
        if "good_weather" not in hist_df.columns and "MeanTemp_C" in hist_df.columns:
            hist_df["good_weather"] = (
                (hist_df["MeanTemp_C"] > 15) & (hist_df["Precipitation_mm"] < 1)
            ).astype(int)

        # Same weekday data
        same_weekdays = hist_df[hist_df["Date"].dt.weekday == pred_weekday].sort_values(
            "Date"
        )

        # Use last known same weekday
        if len(same_weekdays) > 0:
            most_recent_same_day = same_weekdays.iloc[-1]
            most_recent_date = most_recent_same_day["Date"]

            # Lag features
            lagged_features["crew_size_lag_1"] = most_recent_same_day[
                "crew_size_numeric"
            ]
            lagged_features["crew_size_lag_7"] = most_recent_same_day[
                "crew_size_numeric"
            ]
            lagged_features["crew_size_lag_14"] = most_recent_same_day[
                "crew_size_numeric"
            ]
            lagged_features["crew_size_last_week"] = most_recent_same_day[
                "crew_size_numeric"
            ]

            # Rolling windows
            hist_index = hist_df[hist_df["Date"] == most_recent_date].index[0]

            for window in [7, 14]:
                if hist_index >= window - 1:
                    last_days = hist_df.iloc[hist_index - window + 1 : hist_index + 1]

                    # Crew mode and stability
                    crew_mode = last_days["crew_size_numeric"].mode()
                    lagged_features[f"crew_mode_numeric_{window}"] = (
                        crew_mode.iloc[0] if len(crew_mode) > 0 else 3
                    )
                    lagged_features[f"crew_stability_{window}"] = (
                        (
                            last_days["crew_size_numeric"]
                            == lagged_features[f"crew_mode_numeric_{window}"]
                        ).mean()
                        if last_days["crew_size_numeric"].mean() > 0
                        else 0.6
                    )

                    # Weather frequency
                    if "good_weather" in last_days.columns:
                        lagged_features[f"good_weather_freq_{window}"] = last_days[
                            "good_weather"
                        ].mean()

                    # Holiday density
                    holiday_cols = [
                        col for col in last_days.columns if "holiday" in col.lower()
                    ]
                    if holiday_cols:
                        lagged_features[f"holiday_density_{window}"] = (
                            last_days[holiday_cols].any(axis=1).mean()
                        )

            # Weekday crew avg
            lagged_features["weekday_crew_avg"] = same_weekdays[
                "crew_size_numeric"
            ].mean()

        # Default values if missing
        default_values = {
            "crew_size_lag_1": 3,
            "crew_size_lag_7": 3,
            "crew_size_lag_14": 3,
            "crew_size_last_week": 3,
            "crew_mode_numeric_7": 3,
            "crew_mode_numeric_14": 3,
            "crew_stability_7": 0.7,
            "crew_stability_14": 0.6,
            "good_weather_freq_7": 0.3,
            "good_weather_freq_14": 0.3,
            "holiday_density_7": 0.1,
            "holiday_density_14": 0.1,
            "weekday_crew_avg": 3,
        }

        # Apply to input_df
        for feature, default_val in default_values.items():
            if feature in input_df.columns:
                value = lagged_features.get(feature, default_val)
                input_df[feature] = value

        return input_df

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cmap="Blues",
        )
        plt.title("Confusion Matrix - Crew Size Prediction")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    def train(self, df):
        """Improved training with regularization"""
        X, y, feature_cols = self.prepare_data(df)
        self.feature_names = feature_cols

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print("Encoded Classes:", self.label_encoder.classes_)

        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Feature selection to reduce overfitting
        selector = SelectKBest(score_func=f_classif, k=min(20, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        # Store selector for later use
        self.feature_selector = selector
        self.selected_features = [
            feature_cols[i] for i in selector.get_support(indices=True)
        ]
        print(
            f"Selected {len(self.selected_features)} features: {self.selected_features}"
        )

        # More regularized model (reduced complexity)
        self.model = xgb.XGBClassifier(
            n_estimators=200,  # Reduced from 300
            learning_rate=0.08,  # Slightly higher learning rate
            max_depth=3,  # Reduced depth
            subsample=0.8,
            colsample_bytree=0.7,  # Reduced feature sampling
            min_child_weight=5,  # Increased regularization
            reg_alpha=0.2,  # Increased L1 regularization
            reg_lambda=1.5,  # Increased L2 regularization
            random_state=42,
            eval_metric="mlogloss",
        )

        # Simple training without early stopping
        self.model.fit(X_train_selected, y_train)

        # Evaluate
        y_pred_test = self.model.predict(X_test_selected)
        y_pred_train = self.model.predict(X_train_selected)

        y_true_combined = np.concatenate([y_test, y_train])
        y_pred_combined = np.concatenate([y_pred_test, y_pred_train])

        self.plot_confusion_matrix(y_true_combined, y_pred_combined)

        print(f"\nTrain Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        print(
            f"Overfitting Gap: {accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test):.4f}"
        )

        # Print classification report
        print("\nTest Set Classification Report:")
        print(
            classification_report(
                y_test, y_pred_test, target_names=self.label_encoder.classes_
            )
        )

        # Plot feature importance
        self.plot_feature_importance_selected()

        return X_train_selected, X_test_selected, y_train, y_test, y_pred_test

    def plot_feature_importance_selected(self, top_n=15):
        """Plot feature importance for selected features"""
        if self.model is None:
            return

        importance_df = pd.DataFrame(
            {
                "feature": self.selected_features,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(top_n), x="importance", y="feature")
        plt.title("Feature Importance - Selected Features Only")
        plt.tight_layout()
        plt.show()

    def predict(self, df, historical_df=None):
        """Predict with detailed NaN debugging"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("=== PREDICTION NaN DEBUG ===")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input NaN values: {df.isna().sum().sum()}")

        # Engineer features with debugging
        df_features = self.engineer_crew_features(df, historical_df=historical_df)

        print(f"\nAfter feature engineering:")
        print(f"Features DataFrame shape: {df_features.shape}")
        print(f"Features NaN values: {df_features.isna().sum().sum()}")

        print("feature_names needed:", self.feature_names)

        # Check which features are missing
        missing_features = [
            f for f in self.feature_names if f not in df_features.columns
        ]
        if missing_features:
            print(f"\nMISSING FEATURES: {missing_features}")
            return None

        # Select original features
        print(f"\nSelecting features: {len(self.feature_names)} features")
        X = df_features[self.feature_names]

        print(f"Selected features NaN analysis:")
        nan_summary = X.isna().sum()
        total_feature_nan = nan_summary.sum()
        print(f"Total NaN in selected features: {total_feature_nan}")

        if total_feature_nan > 0:
            print("NaN values by feature:")
            for feature, count in nan_summary[nan_summary > 0].items():
                print(f"  {feature}: {count} NaN values")
                # Show some context
                feature_data = X[feature]
                print(f"    Non-NaN count: {feature_data.notna().sum()}")
                if feature_data.notna().sum() > 0:
                    print(
                        f"    Sample values: {feature_data.dropna().head(3).tolist()}"
                    )

        # Handle NaN values systematically
        if total_feature_nan > 0:
            print("\nHandling NaN values...")
            X_cleaned = X.copy()

            # Fill numeric columns with median
            numeric_cols = X_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X_cleaned[col].isna().sum() > 0:
                    median_val = X_cleaned[col].median()
                    if pd.isna(median_val):  # All values are NaN
                        print(f"  {col}: All values NaN, filling with 0")
                        X_cleaned[col] = 0
                    else:
                        print(
                            f"  {col}: Filling {X_cleaned[col].isna().sum()} NaN with median {median_val}"
                        )
                        X_cleaned[col] = X_cleaned[col].fillna(median_val)

            # Fill any remaining columns
            remaining_nan = X_cleaned.isna().sum().sum()
            if remaining_nan > 0:
                print(
                    f"  Filling remaining {remaining_nan} NaN values with forward fill then 0"
                )
                X_cleaned = X_cleaned.fillna(method="ffill").fillna(0)

            X = X_cleaned

        final_nan = X.isna().sum().sum()
        print(f"Final NaN count in X: {final_nan}")

        if final_nan > 0:
            print("ERROR: Still have NaN values after cleaning!")
            return None

        # Scale features
        print("Scaling features...")
        try:
            X_scaled = self.scaler.transform(X)
            print(f"Scaled features shape: {X_scaled.shape}")

            # Check for NaN in scaled features
            scaled_nan_count = np.isnan(X_scaled).sum()
            print(f"NaN in scaled features: {scaled_nan_count}")

            if scaled_nan_count > 0:
                print("Found NaN in scaled features, replacing with 0...")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        except Exception as e:
            print(f"ERROR in scaling: {e}")
            return None

        # Apply feature selection
        print("Applying feature selection...")
        try:
            X_selected = self.feature_selector.transform(X_scaled)
            print(f"Selected features shape: {X_selected.shape}")

            selected_nan_count = np.isnan(X_selected).sum()
            print(f"NaN in selected features: {selected_nan_count}")

            if selected_nan_count > 0:
                print("Found NaN in selected features, replacing with 0...")
                X_selected = np.nan_to_num(X_selected, nan=0.0)

        except Exception as e:
            print(f"ERROR in feature selection: {e}")
            return None

        # Predict
        print("Making prediction...")
        try:
            y_pred_encoded = self.model.predict(X_selected)
            y_pred_proba = self.model.predict_proba(X_selected)

            # Decode predictions
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

            # Create results dataframe
            results = pd.DataFrame(
                {
                    "Date": df["Date"] if "Date" in df.columns else range(len(df)),
                    "predicted_crew_size": y_pred,
                    "prediction_confidence": y_pred_proba.max(axis=1),
                }
            )

            # Add probability for each class
            for i, class_name in enumerate(self.label_encoder.classes_):
                results[f"prob_{class_name}"] = y_pred_proba[:, i]

            print("=== END PREDICTION DEBUG ===\n")
            return results

        except Exception as e:
            print(f"ERROR in prediction: {e}")
            return None

    def analyze_crew_patterns(self, df):
        """Analyze patterns in crew size assignments"""
        df_clean = self.clean_crew_size_data(df)

        # Visitor count vs crew size analysis
        if "Total Visitors_actual" in df.columns:
            crew_visitor_analysis = (
                df_clean.groupby("maat_visitors")
                .agg(
                    {
                        "Total Visitors_actual": [
                            "mean",
                            "median",
                            "std",
                            "min",
                            "max",
                        ],
                        "Date": "count",
                    }
                )
                .round(2)
            )

            print("Visitor Statistics by Crew Size:")
            print(crew_visitor_analysis)

        # Day of week patterns
        if "Date" in df.columns:
            df_clean["Date"] = pd.to_datetime(df_clean["Date"])
            df_clean["day_of_week"] = df_clean["Date"].dt.day_name()

            day_crew_crosstab = (
                pd.crosstab(
                    df_clean["day_of_week"],
                    df_clean["maat_visitors"],
                    normalize="index",
                )
                * 100
            )

            print("\nCrew Size Distribution by Day of Week (%):")
            print(day_crew_crosstab.round(1))

        return crew_visitor_analysis if "Total Visitors_actual" in df.columns else None


if __name__ == "__main__":
    crew_model = CrewSizePredictionModel()

    crew_model.analyze_crew_patterns(df)

    crew_model.train(df)

    crew_model.historical_df = df.copy()

    predictions = crew_model.predict(df, crew_model.historical_df)
    predictions.to_csv(
        "../Data_Sources/Data_Cleaned/Predictions/Crew_Size_Predictions.csv",
        index=False,
    )
