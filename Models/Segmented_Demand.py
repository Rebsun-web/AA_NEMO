import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Read data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modelling_path = os.path.join(
    project_root, "Data_Sources", "Data_Cleaned", "Modelling", "Table_for_modelling.csv"
)

df = pd.read_csv(modelling_path)

# Visitors demand prediction by segments
copy_df = df.copy()
copy_df = copy_df.drop(
    columns=["maat_visitors"]
)  # drop crew predictions, will be predicted later


class SegmentedVisitorPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_sets = {}

        # Define segment mappings (original column names to standardized names)
        self.segment_mappings = {
            "Recreatief NL": "recreatief_nl",
            "Recreatief Buitenland": "recreatief_buitenland",
            "PO": "po",
            "VO": "vo",
            "Student": "student",
            "Extern": "extern",
            "Total Visitors": "total_visitors",
        }

        # Inverse mapping for converting back
        self.inverse_segment_mappings = {v: k for k, v in self.segment_mappings.items()}

    def standardize_column_names(self, df):
        """Standardize column names to snake_case"""
        df = df.copy()

        # Rename segment columns
        rename_dict = {
            old: new for old, new in self.segment_mappings.items() if old in df.columns
        }
        df = df.rename(columns=rename_dict)

        # Standardize other column names
        df.columns = [
            col.lower().replace(" ", "_").replace("/", "_") for col in df.columns
        ]

        return df

    def engineer_features(self, df, target_segment=None):
        """
        Engineer features with standardized column names.
        If target_segment is provided, excludes current segment values
        but keeps historical data.
        """
        df = df.copy()

        # Calculate total visitors if not present
        if "total_visitors" not in df.columns:
            visitor_cols = [
                col
                for col in self.segment_mappings.values()
                if col != "total_visitors" and col in df.columns
            ]
            df["total_visitors"] = df[visitor_cols].sum(axis=1)

        # Get all segment columns except the target
        segment_cols = list(self.segment_mappings.values())
        if target_segment and target_segment in df.columns:
            if target_segment != "total_visitors":
                # Only remove current values of other segments
                current_segments = [
                    col
                    for col in segment_cols
                    if col != target_segment and col != "total_visitors"
                ]
                df = df.drop(columns=current_segments)
            else:
                current_segments = [
                    col for col in segment_cols if col != target_segment
                ]
                df = df.drop(columns=current_segments)

        # Convert date to datetime if not already
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

            # Basic time features
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Season (as one-hot encoded features)
            season_mapping = {
                12: "winter",
                1: "winter",
                2: "winter",
                3: "spring",
                4: "spring",
                5: "spring",
                6: "summer",
                7: "summer",
                8: "summer",
                9: "fall",
                10: "fall",
                11: "fall",
            }
            df["season"] = df["date"].dt.month.map(season_mapping)
            season_dummies = pd.get_dummies(
                df["season"],
                prefix="season",
                drop_first=True,  # Drop one category to avoid multicollinearity
            )
            df = pd.concat([df, season_dummies], axis=1)
            df = df.drop("season", axis=1)

        # Weather features if available
        weather_cols = ["meantemp_c", "precipitation_mm"]
        if all(col in df.columns for col in weather_cols):
            df["good_weather"] = (
                (df["meantemp_c"] > 15) & (df["precipitation_mm"] < 1)
            ).astype(int)
            df["bad_weather"] = (
                (df["meantemp_c"] < 10) | (df["precipitation_mm"] > 5)
            ).astype(int)

            # Bin temperature into categories and one-hot encode
            df["temp_category"] = pd.cut(
                df["meantemp_c"],
                bins=[-float("inf"), 5, 15, 25, float("inf")],
                labels=["cold", "mild", "warm", "hot"],
            )
            temp_dummies = pd.get_dummies(
                df["temp_category"], prefix="temp", drop_first=True
            )
            df = pd.concat([df, temp_dummies], axis=1)
            df = df.drop("temp_category", axis=1)

            # Bin precipitation into categories and one-hot encode
            df["precip_category"] = pd.cut(
                df["precipitation_mm"],
                bins=[-float("inf"), 0.1, 5, float("inf")],
                labels=["dry", "light", "heavy"],
            )
            precip_dummies = pd.get_dummies(
                df["precip_category"], prefix="precip", drop_first=True
            )
            df = pd.concat([df, precip_dummies], axis=1)
            df = df.drop("precip_category", axis=1)

        # Create day type features
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_saturday"] = (df["day_of_week"] == 5).astype(int)
        df["is_sunday"] = (df["day_of_week"] == 6).astype(int)

        # Drop original day_of_week as we have more specific features now
        df = df.drop("day_of_week", axis=1)

        # Add holiday interaction features if available
        holiday_cols = [col for col in df.columns if "holiday" in col.lower()]
        if holiday_cols:
            df["total_holidays"] = df[holiday_cols].sum(axis=1)
            if target_segment == "recreatief_nl":
                df["nl_holiday_effect"] = df["holiday_nl"].fillna(0)
            elif target_segment == "recreatief_buitenland":
                # For international visitors, consider international holidays
                intl_holidays = [col for col in holiday_cols if col != "holiday_nl"]
                df["intl_holiday_effect"] = df[intl_holidays].sum(axis=1)
            elif target_segment in ["po", "vo", "student"]:
                # For educational segments, focus on relevant holidays
                df["edu_holiday_effect"] = (
                    df["holiday_nl"].fillna(0) * 2
                    + df["total_holidays"]  # Double weight for local holidays
                )

        return df

    def add_lagged_features(self, df, segment, lags=[1, 7, 14, 28]):
        """Add lagged features using standardized column names"""
        df = df.copy()

        # Get all segment columns as they represent historical data
        segment_cols = [
            col for col in self.segment_mappings.values() if col in df.columns
        ]

        for lag in lags:
            # Add lags for all segments as they represent historical data
            for seg in segment_cols:
                df[f"{seg}_lag_{lag}"] = df[seg].shift(lag)

            # Add cross-segment features using historical data
            if segment in ["recreatief_nl", "recreatief_buitenland"]:
                rec_cols = ["recreatief_nl", "recreatief_buitenland"]
                rec_cols = [col for col in rec_cols if col in df.columns]
                if rec_cols:
                    df[f"total_recreational_lag_{lag}"] = (
                        df[rec_cols].fillna(0).sum(axis=1).shift(lag)
                    )

            elif segment in ["po", "vo", "student"]:
                edu_cols = ["po", "vo", "student"]
                edu_cols = [col for col in edu_cols if col in df.columns]
                if edu_cols:
                    df[f"total_educational_lag_{lag}"] = (
                        df[edu_cols].fillna(0).sum(axis=1).shift(lag)
                    )

            # Add day-of-week specific lags for the target segment
            if segment in df.columns:
                # Last week same day
                df[f"{segment}_lastweek_sameday"] = df[segment].shift(7)
                # Average of last 4 same weekdays
                df[f"{segment}_avg_4weeks_sameday"] = (
                    df[segment].shift(7)
                    + df[segment].shift(14)
                    + df[segment].shift(21)
                    + df[segment].shift(28)
                ) / 4

        return df

    def add_rolling_features(self, df, segment, windows=[7, 14, 30]):
        """Add rolling features using standardized column names"""
        df = df.copy()

        # Get all segment columns as they represent historical data
        segment_cols = [
            col for col in self.segment_mappings.values() if col in df.columns
        ]

        for window in windows:
            # Add rolling stats for all segments
            for seg in segment_cols:
                df[f"{seg}_rolling_mean_{window}"] = (
                    df[seg].shift(1).rolling(window=window).mean()
                )
                df[f"{seg}_rolling_std_{window}"] = (
                    df[seg].shift(1).rolling(window=window).std()
                )

            # Add segment group rolling features
            if segment in ["recreatief_nl", "recreatief_buitenland"]:
                rec_cols = ["recreatief_nl", "recreatief_buitenland"]
                rec_cols = [col for col in rec_cols if col in df.columns]
                if rec_cols:
                    df[f"total_recreational_rolling_{window}"] = (
                        df[rec_cols]
                        .fillna(0)
                        .sum(axis=1)
                        .shift(1)
                        .rolling(window=window)
                        .mean()
                    )

            elif segment in ["po", "vo", "student"]:
                edu_cols = ["po", "vo", "student"]
                edu_cols = [col for col in edu_cols if col in df.columns]
                if edu_cols:
                    df[f"total_educational_rolling_{window}"] = (
                        df[edu_cols]
                        .fillna(0)
                        .sum(axis=1)
                        .shift(1)
                        .rolling(window=window)
                        .mean()
                    )

            # Add holiday density if available
            if "is_holiday" in df.columns:
                df[f"holiday_density_{window}"] = (
                    df["is_holiday"].rolling(window=window).mean()
                )

        return df

    def prepare_segment_data(self, df, segment):
        """Prepare data for a specific segment using standardized column names"""
        # Standardize column names
        df_processed = self.standardize_column_names(df)

        # Engineer basic features without other segments
        df_processed = self.engineer_features(df_processed, target_segment=segment)

        # Add lagged and rolling features
        df_processed = self.add_lagged_features(df_processed, segment)
        df_processed = self.add_rolling_features(df_processed, segment)

        # Drop rows with NaN values (usually at the start due to lagging)
        df_processed = df_processed.dropna()

        # Drop date column as it's not needed for modeling
        if "date" in df_processed.columns:
            df_processed = df_processed.drop("date", axis=1)

        # Select features and target
        features = [col for col in df_processed.columns if col != segment]
        print(f"Features used for {segment}: {features}")

        X = df_processed[features]
        y = df_processed[segment]

        return X, y, features

    def train_segment_model(self, X, y, segment, visitor_predictions):
        """Train model for a specific segment"""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        # Convert to numpy arrays for scaling
        X_train_scaled = scaler.fit_transform(X_train.to_numpy())
        X_test_scaled = scaler.transform(X_test.to_numpy())

        # Initialize and train model with segment-specific parameters
        if segment == "total_visitors":
            # For total visitors, use a more robust model
            model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=42,
                eval_metric=["rmse", "mae"],
            )
        elif segment == "extern":
            # For extern segment (poorest performing), use more complex model
            model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.005,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=3,
                random_state=42,
                eval_metric=["rmse", "mae"],
            )
        elif segment in ["vo", "student"]:
            # For VO and Student segments (moderate performance)
            model = XGBRegressor(
                n_estimators=750,
                learning_rate=0.008,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.8,
                min_child_weight=2,
                random_state=42,
                eval_metric=["rmse", "mae"],
            )
        else:
            # For better performing segments
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric=["rmse", "mae"],
            )

        # Train the model
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred_test = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)

        # Store predictions with indices
        visitor_predictions[segment] = {
            "train_predictions": pd.Series(y_pred_train, index=X_train.index),
            "test_predictions": pd.Series(y_pred_test, index=X_test.index),
            "train_actual": y_train,
            "test_actual": y_test,
        }

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)

        print(f"\nResults for {self.inverse_segment_mappings[segment]}:")
        print(f"Test_RMSE: {test_rmse:.4f}")
        print(f"Test_R²: {test_r2:.4f}")
        print(f"Test_MAE: {test_mae:.4f}")

        print(f"Train_RMSE: {train_rmse:.4f}")
        print(f"Train_R²: {train_r2:.4f}")
        print(f"Train_MAE: {train_mae:.4f}")

        # Print feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        )
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        ).head(10)

        print("\nTop 10 Most Important Features:")
        for _, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

        return model, scaler, visitor_predictions

    def fit(self, df):
        """Fit models for all segments"""

        # Drop maat_visitors column immediately to avoid conversion issues
        df_clean = df.copy()
        if "maat_visitors" in df_clean.columns:
            df_clean = df_clean.drop(columns=["maat_visitors"])

        # Dictionary to save visitor predictions by segments
        visitor_predictions = {}

        for original_segment, standardized_segment in self.segment_mappings.items():
            print(f"\nTraining model for {original_segment}")
            print("=" * 50)

            # Prepare data
            X, y, features = self.prepare_segment_data(df, standardized_segment)

            # Train model
            model, scaler, visitor_predictions = self.train_segment_model(
                X, y, standardized_segment, visitor_predictions
            )

            # Store model, scaler and features
            self.models[standardized_segment] = model
            self.scalers[standardized_segment] = scaler
            self.feature_sets[standardized_segment] = features

    def predict(self, df, historical_data=None):
        """Make predictions for all segments using historical data for lagged/rolling features"""
        predictions = {}

        # If no historical data provided, we'll need it for lagged/rolling features
        if historical_data is None:
            st.warning(
                "No historical data provided. Lagged and rolling features will be set to 0."
            )

        # Standardize input data column names
        df = self.standardize_column_names(df)

        for original_segment, standardized_segment in self.segment_mappings.items():
            if standardized_segment in self.models:
                # Get expected features from training
                expected_features = self.feature_sets[standardized_segment]

                # Start with the input data
                df_processed = df.copy()

                # Apply basic feature engineering
                df_processed = self.engineer_features(
                    df_processed, target_segment=standardized_segment
                )

                # Calculate lagged and rolling features from historical data
                if historical_data is not None:
                    df_processed = self.add_historical_features(
                        df_processed,
                        historical_data,
                        standardized_segment,
                        expected_features,
                    )

                # Add missing features with default values
                for feature in expected_features:
                    if feature not in df_processed.columns:
                        if (
                            "lag" in feature
                            or "rolling" in feature
                            or "momentum" in feature
                            or "yoy" in feature
                        ):
                            # Use historical averages or set to 0
                            df_processed[feature] = 0
                        elif "season" in feature:
                            # Handle seasonal features based on current date
                            if "date" in df.columns:
                                month = pd.to_datetime(df["date"]).dt.month.iloc[0]
                                if "spring" in feature:
                                    df_processed[feature] = (
                                        1 if month in [3, 4, 5] else 0
                                    )
                                elif "summer" in feature:
                                    df_processed[feature] = (
                                        1 if month in [6, 7, 8] else 0
                                    )
                                elif "winter" in feature:
                                    df_processed[feature] = (
                                        1 if month in [12, 1, 2] else 0
                                    )
                                else:
                                    df_processed[feature] = 0
                            else:
                                df_processed[feature] = 0
                        else:
                            df_processed[feature] = 0

                # Select only the features used in training
                X = df_processed[expected_features]

                # Scale features
                X_scaled = self.scalers[standardized_segment].transform(X)

                # Make predictions
                predictions[original_segment] = self.models[
                    standardized_segment
                ].predict(X_scaled)

        return predictions

    def add_historical_features(
        self, current_data, historical_data, segment, expected_features
    ):
        """Calculate lagged and rolling features using historical data with cutoff handling"""

        # Make a copy and ensure we have a clean index
        current_data = current_data.copy()
        current_data = current_data.reset_index(drop=True)

        # Standardize historical data column names
        hist_data = self.standardize_column_names(historical_data.copy())

        # Check for and handle duplicate columns
        if hist_data.columns.duplicated().any():
            print(
                "Warning: Duplicate columns found in historical data. Removing duplicates."
            )
            # Keep only the first occurrence of each column
            hist_data = hist_data.loc[:, ~hist_data.columns.duplicated()]

        # Ensure date column is datetime
        if "date" in hist_data.columns:
            hist_data["date"] = pd.to_datetime(hist_data["date"])
        if "date" in current_data.columns:
            current_data["date"] = pd.to_datetime(current_data["date"])
            prediction_date = current_data["date"].iloc[0]
        else:
            print("Error: Date column is required for historical feature calculation")
            return current_data

        # Sort historical data by date and get the last available date
        hist_data = hist_data.sort_values("date")
        last_historical_date = hist_data["date"].max()

        print(
            f"Using historical data up to: {last_historical_date.strftime('%Y-%m-%d')}"
        )
        print(f"Predicting for: {prediction_date.strftime('%Y-%m-%d')}")

        # Calculate days between last historical date and prediction date
        days_gap = (prediction_date - last_historical_date).days
        if days_gap > 0:
            print(
                f"Warning: Gap of {days_gap} days between historical data and prediction date"
            )

        # Calculate total visitors if not present in historical data
        if "total_visitors" not in hist_data.columns:
            visitor_cols = [
                col
                for col in self.segment_mappings.values()
                if col != "total_visitors" and col in hist_data.columns
            ]
            if visitor_cols:
                hist_data["total_visitors"] = hist_data[visitor_cols].sum(axis=1)

        # Get all segment columns for feature calculation
        segment_cols = [
            col for col in self.segment_mappings.values() if col in hist_data.columns
        ]

        # Calculate lagged features based on last historical date
        lags = [1, 7, 14, 28]
        for lag in lags:
            # Calculate the target date for this lag from the prediction date
            target_lag_date = prediction_date - pd.Timedelta(days=lag)

            # If the target lag date is after our last historical date,
            # adjust to use the closest available historical data
            if target_lag_date > last_historical_date:
                # Use data from (last_historical_date - remaining_lag_days)
                remaining_lag = lag - days_gap
                if remaining_lag > 0:
                    adjusted_lag_date = last_historical_date - pd.Timedelta(
                        days=remaining_lag
                    )
                else:
                    # If even adjusted lag goes beyond our data, use the last available data
                    adjusted_lag_date = last_historical_date
            else:
                adjusted_lag_date = target_lag_date

            # Find the closest historical date
            closest_idx = None
            if len(hist_data.loc[hist_data["date"] <= adjusted_lag_date]) > 0:
                closest_idx = hist_data.loc[
                    hist_data["date"] <= adjusted_lag_date, "date"
                ].idxmax()

            if closest_idx is not None:
                # Add lags for all segments
                for seg in segment_cols:
                    feature_name = f"{seg}_lag_{lag}"
                    if feature_name in expected_features:
                        if seg in hist_data.columns:
                            value = hist_data.loc[closest_idx, seg]
                            if isinstance(value, pd.Series):
                                value = value.iloc[0]
                        else:
                            value = 0

                        # Assign value to all rows in current_data
                        current_data.loc[:, feature_name] = value

                # Add cross-segment features
                if segment in ["recreatief_nl", "recreatief_buitenland"]:
                    rec_cols = ["recreatief_nl", "recreatief_buitenland"]
                    rec_cols = [col for col in rec_cols if col in hist_data.columns]
                    feature_name = f"total_recreational_lag_{lag}"
                    if rec_cols and feature_name in expected_features:
                        value = hist_data.loc[closest_idx, rec_cols].sum()
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        current_data.loc[:, feature_name] = value

                elif segment in ["po", "vo", "student"]:
                    edu_cols = ["po", "vo", "student"]
                    edu_cols = [col for col in edu_cols if col in hist_data.columns]
                    feature_name = f"total_educational_lag_{lag}"
                    if edu_cols and feature_name in expected_features:
                        value = hist_data.loc[closest_idx, edu_cols].sum()
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        current_data.loc[:, feature_name] = value

                # Add day-of-week specific lags
                if segment in hist_data.columns:
                    if f"{segment}_lastweek_sameday" in expected_features:
                        # For last week same day, adjust similarly
                        target_same_day = prediction_date - pd.Timedelta(days=7)
                        if target_same_day > last_historical_date:
                            # Use the last available same weekday
                            prediction_weekday = prediction_date.weekday()
                            same_weekday_data = hist_data[
                                hist_data["date"].dt.weekday == prediction_weekday
                            ]
                            if len(same_weekday_data) > 0:
                                last_same_weekday_idx = same_weekday_data[
                                    "date"
                                ].idxmax()
                                value = hist_data.loc[last_same_weekday_idx, segment]
                                if isinstance(value, pd.Series):
                                    value = value.iloc[0]
                                current_data.loc[
                                    :, f"{segment}_lastweek_sameday"
                                ] = value
                        else:
                            closest_same_day = hist_data.loc[
                                hist_data["date"] <= target_same_day, "date"
                            ].idxmax()
                            value = hist_data.loc[closest_same_day, segment]
                            if isinstance(value, pd.Series):
                                value = value.iloc[0]
                            current_data.loc[:, f"{segment}_lastweek_sameday"] = value

                    # Calculate average of last 4 same weekdays
                    if f"{segment}_avg_4weeks_sameday" in expected_features:
                        prediction_weekday = prediction_date.weekday()
                        same_weekday_dates = [
                            prediction_date - pd.Timedelta(days=7),
                            prediction_date - pd.Timedelta(days=14),
                            prediction_date - pd.Timedelta(days=21),
                            prediction_date - pd.Timedelta(days=28),
                        ]

                        same_weekday_values = []
                        for target_date in same_weekday_dates:
                            if target_date <= last_historical_date:
                                # Use exact or closest date
                                closest = (
                                    hist_data.loc[
                                        hist_data["date"] <= target_date, "date"
                                    ].idxmax()
                                    if len(
                                        hist_data.loc[hist_data["date"] <= target_date]
                                    )
                                    > 0
                                    else None
                                )
                                if closest is not None and segment in hist_data.columns:
                                    value = hist_data.loc[closest, segment]
                                    if isinstance(value, pd.Series):
                                        value = value.iloc[0]
                                    same_weekday_values.append(value)
                            else:
                                # Use historical same weekdays
                                same_weekday_historical = hist_data[
                                    hist_data["date"].dt.weekday == prediction_weekday
                                ]
                                if len(same_weekday_historical) > 0:
                                    # Use the most recent same weekday
                                    recent_same_weekday = same_weekday_historical.loc[
                                        same_weekday_historical["date"].idxmax(),
                                        segment,
                                    ]
                                    if isinstance(recent_same_weekday, pd.Series):
                                        recent_same_weekday = recent_same_weekday.iloc[
                                            0
                                        ]
                                    same_weekday_values.append(recent_same_weekday)

                        if same_weekday_values:
                            current_data.loc[
                                :, f"{segment}_avg_4weeks_sameday"
                            ] = np.mean(same_weekday_values)
            else:
                # If no historical data available for this lag, use the most recent available data
                if len(hist_data) > 0:
                    last_idx = hist_data["date"].idxmax()
                    for seg in segment_cols:
                        feature_name = f"{seg}_lag_{lag}"
                        if feature_name in expected_features:
                            value = (
                                hist_data.loc[last_idx, seg]
                                if seg in hist_data.columns
                                else 0
                            )
                            if isinstance(value, pd.Series):
                                value = value.iloc[0]
                            current_data.loc[:, feature_name] = value

        # Calculate rolling features based on available historical data
        windows = [7, 14, 30]
        for window in windows:
            # Calculate the target window start date
            target_window_start = prediction_date - pd.Timedelta(days=window)

            # Adjust window to use available historical data
            if target_window_start > last_historical_date:
                # Use the last available window of the same size
                adjusted_window_start = last_historical_date - pd.Timedelta(days=window)
                adjusted_window_end = last_historical_date
            else:
                # Use data up to the last historical date, but maintain window size
                adjusted_window_end = min(
                    last_historical_date, prediction_date - pd.Timedelta(days=1)
                )
                adjusted_window_start = adjusted_window_end - pd.Timedelta(days=window)

            # Get window data
            window_data = hist_data[
                (hist_data["date"] >= adjusted_window_start)
                & (hist_data["date"] <= adjusted_window_end)
            ]

            if len(window_data) > 0:
                # Add rolling stats for all segments
                for seg in segment_cols:
                    feature_mean = f"{seg}_rolling_mean_{window}"
                    feature_std = f"{seg}_rolling_std_{window}"

                    if feature_mean in expected_features and seg in window_data.columns:
                        mean_value = float(window_data[seg].mean())
                        current_data.loc[:, feature_mean] = mean_value

                    if feature_std in expected_features and seg in window_data.columns:
                        std_value = (
                            float(window_data[seg].std())
                            if len(window_data) > 1
                            else 0.0
                        )
                        current_data.loc[:, feature_std] = std_value

                # Add segment group rolling features
                if segment in ["recreatief_nl", "recreatief_buitenland"]:
                    rec_cols = ["recreatief_nl", "recreatief_buitenland"]
                    rec_cols = [col for col in rec_cols if col in window_data.columns]
                    feature_name = f"total_recreational_rolling_{window}"
                    if rec_cols and feature_name in expected_features:
                        value = float(window_data[rec_cols].sum(axis=1).mean())
                        current_data.loc[:, feature_name] = value

                elif segment in ["po", "vo", "student"]:
                    edu_cols = ["po", "vo", "student"]
                    edu_cols = [col for col in edu_cols if col in window_data.columns]
                    feature_name = f"total_educational_rolling_{window}"
                    if edu_cols and feature_name in expected_features:
                        value = float(window_data[edu_cols].sum(axis=1).mean())
                        current_data.loc[:, feature_name] = value
            else:
                # If no data available for this window, use the last available values
                if len(hist_data) > 0:
                    last_values = hist_data.tail(min(window, len(hist_data)))
                    for seg in segment_cols:
                        feature_mean = f"{seg}_rolling_mean_{window}"
                        feature_std = f"{seg}_rolling_std_{window}"

                        if (
                            feature_mean in expected_features
                            and seg in last_values.columns
                        ):
                            # Ensure we get a single column even if there are duplicates
                            seg_values = last_values[seg]
                            if isinstance(seg_values, pd.DataFrame):
                                # Multiple columns with same name - take first
                                seg_values = seg_values.iloc[:, 0]

                            # Calculate mean and ensure it's a scalar
                            mean_value = seg_values.mean()
                            if isinstance(mean_value, pd.Series):
                                mean_value = mean_value.iloc[0]

                            current_data.loc[:, feature_mean] = float(mean_value)

                        if (
                            feature_std in expected_features
                            and seg in last_values.columns
                        ):
                            # Ensure we get a single column even if there are duplicates
                            seg_values = last_values[seg]
                            if isinstance(seg_values, pd.DataFrame):
                                # Multiple columns with same name - take first
                                seg_values = seg_values.iloc[:, 0]

                            # Calculate std and ensure it's a scalar
                            if len(seg_values) > 1:
                                std_value = seg_values.std()
                                if isinstance(std_value, pd.Series):
                                    std_value = std_value.iloc[0]
                                std_value = float(std_value)
                            else:
                                std_value = 0.0

                            current_data.loc[:, feature_std] = std_value

        return current_data

    def get_predictions_dataframe(self, df):
        """
        Create a DataFrame with predictions for all segments, preserving all original rows
        """
        # First, fit the models if not already done
        if not self.models:
            self.fit(df)

        # Initialize results with full original index
        all_predictions = {}
        original_index = df.index

        # Initialize all columns with NaN
        for original_segment in self.segment_mappings.keys():
            all_predictions[f"{original_segment}_pred"] = pd.Series(
                index=original_index, dtype=float
            )
            all_predictions[f"{original_segment}_actual"] = pd.Series(
                index=original_index, dtype=float
            )
            all_predictions[f"{original_segment}_set"] = pd.Series(
                index=original_index, dtype=str
            )

        # Get predictions from each model for rows that can be processed
        for segment in self.segment_mappings.values():
            if segment in self.models:
                # Re-prepare data to get the same train/test split
                X, y, features = self.prepare_segment_data(df, segment)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Get predictions
                X_train_scaled = self.scalers[segment].transform(X_train.to_numpy())
                X_test_scaled = self.scalers[segment].transform(X_test.to_numpy())

                y_pred_train = self.models[segment].predict(X_train_scaled)
                y_pred_test = self.models[segment].predict(X_test_scaled)

                original_name = self.inverse_segment_mappings[segment]

                # Fill in train predictions and labels
                all_predictions[f"{original_name}_pred"].loc[
                    X_train.index
                ] = y_pred_train
                all_predictions[f"{original_name}_actual"].loc[
                    X_train.index
                ] = y_train.values
                all_predictions[f"{original_name}_set"].loc[X_train.index] = "train"

                # Fill in test predictions and labels
                all_predictions[f"{original_name}_pred"].loc[X_test.index] = y_pred_test
                all_predictions[f"{original_name}_actual"].loc[
                    X_test.index
                ] = y_test.values
                all_predictions[f"{original_name}_set"].loc[X_test.index] = "test"

        # Create DataFrame
        predictions_df = pd.DataFrame(all_predictions)

        # Add date column
        if "Date" in df.columns:
            predictions_df["Date"] = df["Date"]
        elif "date" in df.columns:
            predictions_df["Date"] = df["date"]

        return predictions_df

    def analyze_feature_importance(self, segment=None, top_n=15, plot=True):
        """
        Analyze feature importance for a specific segment or all segments.

        Args:
            segment (str, optional): Specific segment to analyze.
                                   If None, analyzes all segments.
            top_n (int): Number of top features to show.
            plot (bool): Whether to create visualizations.

        Returns:
            dict: Dictionary containing feature importance analysis for each segment.
        """
        results = {}
        segments_to_analyze = (
            [segment] if segment else list(self.segment_mappings.values())
        )

        for seg in segments_to_analyze:
            if seg not in self.models:
                continue

            # Get feature importance
            importance = self.models[seg].feature_importances_
            features = self.feature_sets[seg]

            # Create importance DataFrame
            imp_df = pd.DataFrame({"feature": features, "importance": importance})
            imp_df = imp_df.sort_values("importance", ascending=False)

            # Group features by type
            feature_types = {
                "lag": "Historical Values",
                "rolling": "Rolling Statistics",
                "season": "Seasonal",
                "temp": "Temperature",
                "precip": "Precipitation",
                "holiday": "Holidays",
                "is_": "Day Type",
            }

            def get_feature_type(feature_name):
                for key, value in feature_types.items():
                    if key in feature_name:
                        return value
                return "Other"

            imp_df["feature_type"] = imp_df["feature"].apply(get_feature_type)

            # Calculate type importance
            type_importance = (
                imp_df.groupby("feature_type")["importance"]
                .sum()
                .sort_values(ascending=False)
            )

            # Store results
            results[self.inverse_segment_mappings[seg]] = {
                "top_features": imp_df.head(top_n),
                "feature_type_importance": type_importance,
            }

            if plot:
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(
                    f"Feature Importance Analysis - {self.inverse_segment_mappings[seg]}",
                    fontsize=14,
                )

                # Plot top features
                sns.barplot(
                    data=imp_df.head(top_n), x="importance", y="feature", ax=ax1
                )
                ax1.set_title("Top Individual Features")
                ax1.set_xlabel("Importance Score")

                # Plot feature type importance
                sns.barplot(x=type_importance.values, y=type_importance.index, ax=ax2)
                ax2.set_title("Feature Type Importance")
                ax2.set_xlabel("Total Importance Score")

                plt.tight_layout()
                plt.show()

        return results

    def get_segment_insights(self, segment):
        """
        Get detailed insights about what influences a specific segment.

        Args:
            segment (str): The segment to analyze

        Returns:
            dict: Dictionary containing insights about the segment
        """
        if segment not in self.models:
            return None

        # Get standardized segment name
        std_segment = segment
        if segment in self.inverse_segment_mappings:
            std_segment = segment
        elif segment in self.segment_mappings:
            std_segment = self.segment_mappings[segment]
        else:
            return None

        # Get feature importance analysis
        analysis = self.analyze_feature_importance(segment=std_segment, plot=False)
        segment_analysis = analysis[self.inverse_segment_mappings[std_segment]]

        # Extract key insights
        top_features = segment_analysis["top_features"]
        type_importance = segment_analysis["feature_type_importance"]

        # Generate insights
        insights = {
            "segment": self.inverse_segment_mappings[std_segment],
            "top_5_features": top_features.head().to_dict("records"),
            "main_drivers": type_importance.head(3).to_dict(),
            "recommendations": [],
        }

        # Add specific recommendations based on feature importance
        if "Historical Values" in type_importance.head(3):
            insights["recommendations"].append(
                "Strong dependence on historical patterns - "
                "consider recent trends for predictions"
            )

        if "Seasonal" in type_importance.head(3):
            insights["recommendations"].append(
                "Seasonal factors are important - " "plan for seasonal variations"
            )

        if "Holidays" in type_importance.head(3):
            insights["recommendations"].append(
                "Holiday periods significantly impact visitors - "
                "adjust staffing during holidays"
            )

        if "Temperature" in type_importance.head(3):
            insights["recommendations"].append(
                "Weather sensitive segment - " "consider weather forecasts in planning"
            )

        return insights


if __name__ == "__main__":
    predictor = SegmentedVisitorPredictor()

    # Train on your data
    predictor.fit(copy_df)

    # Make predictions
    predictions = predictor.predict(copy_df)

    # Get Predictions DataFrame
    predictions_df = predictor.get_predictions_dataframe(copy_df)

    # The most important features
    predictor.analyze_feature_importance()
    predictor.get_segment_insights("Recreatief NL")

    # Save Visitors Demand Prediction
    try:
        os.makedirs("../Data_Sources/Data_Cleaned/Predictions", exist_ok=True)

        path = "../Data_Sources/Data_Cleaned/Predictions/Segmented_Visitor_Demand_Prediction.csv"
        predictions_df.to_csv(path, index=False)
        print(f"Predictions saved successfully to: {path}")

    except Exception as e:
        print(f"Error saving file: {e}")
