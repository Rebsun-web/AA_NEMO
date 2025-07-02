import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.patches import Patch

# --- Data Loading ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modelling_path = os.path.join(
    project_root,
    "Data_Sources",
    "Data_Modelling",
    "Modelling",
    "Table_for_modelling.csv"
)
df = pd.read_csv(modelling_path)

# --- Prepare Data ---
copy_df = df.copy()
copy_df = copy_df.drop(columns=["maat_visitors"])  # drop crew predictions, will be predicted later


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
        self.inverse_segment_mappings = {v: k for k, v in self.segment_mappings.items()}

    def format_feature_name(self, feature_name):
        """
        Format feature names for better readability in plots.
        Converts snake_case to Title Case and handles special formatting.
        """
        # Replace underscores with spaces
        formatted = feature_name.replace('_', ' ')
        
        # Handle special cases for common abbreviations and terms
        replacements = {
            ' nl ': ' NL ',
            ' vo ': ' VO ',
            ' po ': ' PO ',
            ' pv ': ' PV ',
            ' yoy ': ' YoY ',
            ' cos ': ' Cos ',
            ' sin ': ' Sin ',
            'lag ': 'Lag ',
            'momentum': 'Momentum',
            'occupancy': 'Occupancy',
            'buitenland': 'Buitenland',
            'recreatief': 'Recreational',
            'recreationeel': 'Recreational',
            'student': 'Student',
            'extern': 'Extern',
            'holiday': 'Holiday',
            'weekend': 'Weekend',
            'theaters': 'Theaters',
            'museums': 'Museums',
            'hotel': 'Hotel',
            'belgium': 'Belgium',
            'france': 'France',
            'italy': 'Italy',
            'holland': 'Holland',
            'meantemp': 'Meantemp',
            'dayofweek': 'Dayofweek',
            'duration': 'Duration',
            'minutes': 'Minutes',
            'disruptions': 'Disruptions',
            'count': 'Count',
            'index': 'Index',
            'strength': 'Strength',
            'season': 'Season',
            'shoulder': 'Shoulder',
            'tourism': 'Tourism',
            'peak': 'Peak',
            'summer': 'Summer',
            'very': 'Very',
            'hot': 'Hot',
            'open': 'Open',
            'nemo': 'Nemo',
            'north': 'North'
        }
        
        # Apply replacements (case insensitive)
        formatted_lower = formatted.lower()
        for old, new in replacements.items():
            formatted_lower = formatted_lower.replace(old.lower(), f' {new} ')
        
        # Clean up extra spaces and capitalize appropriately
        words = formatted_lower.split()
        formatted_words = []
        
        for word in words:
            word = word.strip()
            if word:
                # Keep certain words in specific cases
                if word.upper() in ['NL', 'VO', 'PO', 'PV', 'YOY']:
                    formatted_words.append(word.upper())
                elif word in ['Lag', 'Momentum', 'Cos', 'Sin']:
                    formatted_words.append(word)
                else:
                    formatted_words.append(word.capitalize())
        
        # Convert "Lag X" to "(Lag X)" format
        result = ' '.join(formatted_words)
        result = re.sub(r'\bLag (\d+)\b', r'(Lag \1)', result)
        
        return result

    def plot_segment_correlations(self, df, segment, exclude_columns=None):
        """
        Plot feature correlations with each segment's visitor count.
        Shows top 20 positive and top 20 negative correlations for each segment.
        
        Args:
            df (pd.DataFrame): The dataframe with all features and segment columns.
            segment (str): The segment column to analyze correlations for.
            exclude_columns (list): Columns to exclude from correlation analysis.
        """
        if exclude_columns is None:
            exclude_columns = []
        
        segment_columns = [
            "recreatief_nl", "recreatief_buitenland", "po", "vo", "student", "extern", "total_visitors"
        ]
        
        # Only use numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col not in segment_columns + exclude_columns]
        
        # Compute correlations
        corr = df[features + [segment]].corr()[segment].drop(segment)
        
        # Select top 20 positive and top 20 negative correlations
        top_pos = corr.sort_values(ascending=False).head(20)
        top_neg = corr.sort_values().head(20)
        corr_to_plot = pd.concat([top_pos, top_neg]).drop_duplicates().sort_values()
        
        # Format feature names for display
        formatted_names = [self.format_feature_name(f) for f in corr_to_plot.index]
        
        # Identify features with "(Lag)" in their formatted names for green coloring
        colors = []
        for formatted_name in formatted_names:
            if "(Lag" in formatted_name or "Rolling" in formatted_name:
                colors.append("green")
            else:
                colors.append("black")
        
        # Plot
        plt.figure(figsize=(18, 10))
        bars = plt.barh(formatted_names, corr_to_plot.values, color=colors)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        
        # Format title
        segment_title = self.format_feature_name(segment)
        plt.title(f"Top 20 Positive & Negative Feature Correlations with {segment_title}", 
                fontsize=14, fontweight='bold')
        plt.xlabel("Correlation Coefficient", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        
        # Legend
        legend_elements = [
            Patch(facecolor='black', label='Regular Features'),
            Patch(facecolor='green', label='Lagged Features')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        # Improve layout
        plt.tight_layout()
        plt.grid(axis='x', alpha=0.3)
        
        # Rotate y-axis labels if they're too long
        plt.tick_params(axis='y', labelsize=10)
        
        plt.show()

    def standardize_column_names(self, df):
        """Standardize column names to snake_case and segment names."""
        df = df.copy()
        rename_dict = {old: new for old, new in self.segment_mappings.items() if old in df.columns}
        df = df.rename(columns=rename_dict)
        df.columns = [col.lower().replace(" ", "_").replace("/", "_") for col in df.columns]
        return df

    def engineer_features(self, df, target_segment=None):
        """
        Engineer features with standardized column names.
        If target_segment is provided, excludes current segment values but keeps historical data.
        """
        df = df.copy()

        # Calculate total visitors if not present
        if "total_visitors" not in df.columns:
            visitor_cols = [col for col in self.segment_mappings.values() if col != "total_visitors" and col in df.columns]
            df["total_visitors"] = df[visitor_cols].sum(axis=1)

        # Remove current values of other segments for the target
        segment_cols = list(self.segment_mappings.values())
        if target_segment and target_segment in df.columns:
            if target_segment != "total_visitors":
                current_segments = [col for col in segment_cols if col != target_segment and col != "total_visitors"]
                df = df.drop(columns=current_segments)
            else:
                current_segments = [col for col in segment_cols if col != target_segment]
                df = df.drop(columns=current_segments)

        # Time features
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            season_mapping = {
                12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "fall", 10: "fall", 11: "fall"
            }
            df["season"] = df["date"].dt.month.map(season_mapping)
            season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=True, dtype=int)
            df = pd.concat([df, season_dummies], axis=1)
            df = df.drop("season", axis=1)

        # Weather features
        weather_cols = ["meantemp_c", "precipitation_mm"]
        if all(col in df.columns for col in weather_cols):
            df["good_weather"] = ((df["meantemp_c"] > 15) & (df["precipitation_mm"] < 1)).astype(int)
            df["bad_weather"] = ((df["meantemp_c"] < 10) | (df["precipitation_mm"] > 5)).astype(int)
            df["temp_category"] = pd.cut(
                df["meantemp_c"],
                bins=[-float("inf"), 5, 15, 25, float("inf")],
                labels=["cold", "mild", "warm", "hot"]
            )
            temp_dummies = pd.get_dummies(df["temp_category"], prefix="temp", drop_first=True, dtype=int)
            df = pd.concat([df, temp_dummies], axis=1)
            df = df.drop("temp_category", axis=1)
            df["precip_category"] = pd.cut(
                df["precipitation_mm"],
                bins=[-float("inf"), 0.1, 5, float("inf")],
                labels=["dry", "light", "heavy"]
            )
            precip_dummies = pd.get_dummies(df["precip_category"], prefix="precip", drop_first=True, dtype=int)
            df = pd.concat([df, precip_dummies], axis=1)
            df = df.drop("precip_category", axis=1)

        # Day type features
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_saturday"] = (df["day_of_week"] == 5).astype(int)
        df["is_sunday"] = (df["day_of_week"] == 6).astype(int)
        df = df.drop("day_of_week", axis=1)

        # Holiday features
        holiday_cols = [col for col in df.columns if "holiday" in col.lower()]
        if holiday_cols:
            df["total_holidays"] = df[holiday_cols].sum(axis=1)
            if target_segment == "recreatief_nl":
                df["nl_holiday_effect"] = df["public_holiday"].fillna(0)
            elif target_segment == "recreatief_buitenland":
                intl_holidays = [col for col in holiday_cols if col != "public_holiday"]
                df["intl_holiday_effect"] = df[intl_holidays].sum(axis=1)
            elif target_segment in ["po", "vo", "student"]:
                df["edu_holiday_effect"] = df["school_holiday"].fillna(0) * 2 + df["total_holidays"]

        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        # Handle remaining object columns by converting to category codes or dummy variables
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col != 'date':  # Skip date column
                # If few unique values, create dummy variables
                if df[col].nunique() <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(col, axis=1)
                else:
                    # For high cardinality, use label encoding
                    df[col] = pd.Categorical(df[col]).codes

        return df

    def add_lagged_features(self, df, segment, lags=[1, 7, 14, 28]):
        df = df.copy()
        segment_cols = [col for col in self.segment_mappings.values() if col in df.columns]
        for lag in lags:
            for seg in segment_cols:
                df[f"{seg}_lag_{lag}"] = df[seg].shift(lag)
            if segment in ["recreatief_nl", "recreatief_buitenland"]:
                rec_cols = [col for col in ["recreatief_nl", "recreatief_buitenland"] if col in df.columns]
                if rec_cols:
                    df[f"total_recreational_lag_{lag}"] = df[rec_cols].fillna(0).sum(axis=1).shift(lag)
            elif segment in ["po", "vo", "student"]:
                edu_cols = [col for col in ["po", "vo", "student"] if col in df.columns]
                if edu_cols:
                    df[f"total_educational_lag_{lag}"] = df[edu_cols].fillna(0).sum(axis=1).shift(lag)
            if segment in df.columns:
                df[f"{segment}_lastweek_sameday"] = df[segment].shift(7)
                df[f"{segment}_avg_4weeks_sameday"] = (
                    df[segment].shift(7) + df[segment].shift(14) + df[segment].shift(21) + df[segment].shift(28)
                ) / 4
        return df

    def add_rolling_features(self, df, segment, windows=[7, 14, 30]):
        df = df.copy()
        segment_cols = [col for col in self.segment_mappings.values() if col in df.columns]
        for window in windows:
            for seg in segment_cols:
                df[f"{seg}_rolling_mean_{window}"] = df[seg].shift(1).rolling(window=window).mean()
                df[f"{seg}_rolling_std_{window}"] = df[seg].shift(1).rolling(window=window).std()
            if segment in ["recreatief_nl", "recreatief_buitenland"]:
                rec_cols = [col for col in ["recreatief_nl", "recreatief_buitenland"] if col in df.columns]
                if rec_cols:
                    df[f"total_recreational_rolling_{window}"] = df[rec_cols].fillna(0).sum(axis=1).shift(1).rolling(window=window).mean()
            elif segment in ["po", "vo", "student"]:
                edu_cols = [col for col in ["po", "vo", "student"] if col in df.columns]
                if edu_cols:
                    df[f"total_educational_rolling_{window}"] = df[edu_cols].fillna(0).sum(axis=1).shift(1).rolling(window=window).mean()
            if "is_holiday" in df.columns:
                df[f"holiday_density_{window}"] = df["is_holiday"].rolling(window=window).mean()
        return df

    def prepare_segment_data(self, df, segment):
        df_processed = self.standardize_column_names(df)
        df_processed = self.engineer_features(df_processed, target_segment=segment)
        df_processed = self.add_lagged_features(df_processed, segment)
        df_processed = self.add_rolling_features(df_processed, segment)
        
        # Only drop rows where the target variable is NaN
        df_processed = df_processed.dropna(subset=[segment])
        
        # Fill lagged features with forward fill or mean
        lag_cols = [col for col in df_processed.columns if 'lag' in col or 'rolling' in col]
        for col in lag_cols:
            if col in df_processed.columns:
                # Use the newer pandas methods instead of deprecated ones
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        
        # Fill other numerical columns with their median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Fill categorical columns with mode or 0
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0)
        
        # Remove non-numeric columns and other columns that shouldn't be features
        columns_to_exclude = [
            "date", "year", "month",  # Time-related
            "has_predictions", "prediction_made_date",  # Metadata columns
            segment  # Target variable
        ]
        
        # Also exclude any remaining string/object columns that aren't dummy variables
        # string_cols = df_processed.select_dtypes(include=['object', 'datetime', 'datetime64']).columns
        # columns_to_exclude.extend(string_cols.tolist())
        
        # Remove duplicates and ensure target is excluded
        columns_to_exclude = list(set(columns_to_exclude))
        
        # Get all columns except those to exclude
        all_columns = df_processed.columns.tolist()
        features = [col for col in all_columns if col not in columns_to_exclude]
        
        print(f"Features used for {segment}: {len(features)} features")
        print(f"Dataset size after processing: {len(df_processed)} rows")
        print(f"Excluded columns: {[col for col in columns_to_exclude if col in all_columns]}")
        
        # Check if we have any data left
        if len(df_processed) == 0:
            print(f"ERROR: No data remaining for segment {segment} after processing!")
            return None, None, None
        
        # Double-check that all feature columns are numeric
        X = df_processed[features]
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"WARNING: Non-numeric columns found in features: {non_numeric.tolist()}")
            # Remove these columns
            features = [col for col in features if col not in non_numeric]
            # X = df_processed[features]
            print(f"Updated feature count after removing non-numeric: {len(features)}")
        
        # Optionally, exclude columns like 'date', etc.
        exclude_columns = ["year", "month", "has_predictions", "prediction_made_date"]
        self.plot_segment_correlations(df_processed, segment, exclude_columns)

        y = df_processed[segment]
        return X, y, features

    def train_segment_model(self, X, y, segment, visitor_predictions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.to_numpy())
        X_test_scaled = scaler.transform(X_test.to_numpy())
        if segment == "total_visitors":
            model = XGBRegressor(
                n_estimators=1000, learning_rate=0.01, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                random_state=42, eval_metric=["rmse", "mae"]
            )
        elif segment == "extern":
            model = XGBRegressor(
                n_estimators=1000, learning_rate=0.005, max_depth=8,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                random_state=42, eval_metric=["rmse", "mae"]
            )
        elif segment in ["vo", "student"]:
            model = XGBRegressor(
                n_estimators=750, learning_rate=0.008, max_depth=6,
                subsample=0.85, colsample_bytree=0.8, min_child_weight=2,
                random_state=42, eval_metric=["rmse", "mae"]
            )
        else:
            model = XGBRegressor(
                n_estimators=500, learning_rate=0.01, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric=["rmse", "mae"]
            )
        model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)], verbose=False)
        y_pred_test = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)
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
        feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        feature_importance = feature_importance.sort_values("importance", ascending=False).head(10)
        print("\nTop 10 Most Important Features:")
        for _, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        return model, scaler, visitor_predictions

    def fit(self, df):
        df_clean = df.copy()
        if "maat_visitors" in df_clean.columns:
            df_clean = df_clean.drop(columns=["maat_visitors"])
        visitor_predictions = {}
        for original_segment, standardized_segment in self.segment_mappings.items():
            print(f"\nTraining model for {original_segment}")
            print("=" * 50)
            X, y, features = self.prepare_segment_data(df, standardized_segment)
            model, scaler, visitor_predictions = self.train_segment_model(X, y, standardized_segment, visitor_predictions)
            self.models[standardized_segment] = model
            self.scalers[standardized_segment] = scaler
            self.feature_sets[standardized_segment] = features

    def predict(self, df, historical_data=None):
        """Make predictions for all segments using historical data for lagged/rolling features"""
        predictions = {}

        # If no historical data provided, we'll need it for lagged/rolling features
        if historical_data is None:
            print(
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

    def add_historical_features(self, current_data, historical_data, segment, expected_features):
        """
        Calculate lagged and rolling features using historical data with cutoff handling.
        """
        # Make a copy and ensure we have a clean index
        current_data = current_data.copy().reset_index(drop=True)

        # Standardize historical data column names
        hist_data = self.standardize_column_names(historical_data.copy())

        # Check for and handle duplicate columns
        if hist_data.columns.duplicated().any():
            print("Warning: Duplicate columns found in historical data. Removing duplicates.")
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

        print(f"Using historical data up to: {last_historical_date.strftime('%Y-%m-%d')}")
        print(f"Predicting for: {prediction_date.strftime('%Y-%m-%d')}")

        # Calculate days between last historical date and prediction date
        days_gap = (prediction_date - last_historical_date).days
        if days_gap > 0:
            print(f"Warning: Gap of {days_gap} days between historical data and prediction date")

        # Calculate total visitors if not present in historical data
        if "total_visitors" not in hist_data.columns:
            visitor_cols = [
                col for col in self.segment_mappings.values()
                if col != "total_visitors" and col in hist_data.columns
            ]
            if visitor_cols:
                hist_data["total_visitors"] = hist_data[visitor_cols].sum(axis=1)

        # Get all segment columns for feature calculation
        segment_cols = [col for col in self.segment_mappings.values() if col in hist_data.columns]

        # Calculate lagged features based on last historical date
        lags = [1, 7, 14, 28]
        for lag in lags:
            target_lag_date = prediction_date - pd.Timedelta(days=lag)
            if target_lag_date > last_historical_date:
                remaining_lag = lag - days_gap
                adjusted_lag_date = last_historical_date - pd.Timedelta(days=remaining_lag) if remaining_lag > 0 else last_historical_date
            else:
                adjusted_lag_date = target_lag_date

            closest_idx = None
            if len(hist_data.loc[hist_data["date"] <= adjusted_lag_date]) > 0:
                closest_idx = hist_data.loc[hist_data["date"] <= adjusted_lag_date, "date"].idxmax()

            if closest_idx is not None:
                # Add lags for all segments
                for seg in segment_cols:
                    feature_name = f"{seg}_lag_{lag}"
                    if feature_name in expected_features:
                        value = hist_data.loc[closest_idx, seg] if seg in hist_data.columns else 0
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        current_data.loc[:, feature_name] = value

                # Add cross-segment features
                if segment in ["recreatief_nl", "recreatief_buitenland"]:
                    rec_cols = [col for col in ["recreatief_nl", "recreatief_buitenland"] if col in hist_data.columns]
                    feature_name = f"total_recreational_lag_{lag}"
                    if rec_cols and feature_name in expected_features:
                        value = hist_data.loc[closest_idx, rec_cols].sum()
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        current_data.loc[:, feature_name] = value

                elif segment in ["po", "vo", "student"]:
                    edu_cols = [col for col in ["po", "vo", "student"] if col in hist_data.columns]
                    feature_name = f"total_educational_lag_{lag}"
                    if edu_cols and feature_name in expected_features:
                        value = hist_data.loc[closest_idx, edu_cols].sum()
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        current_data.loc[:, feature_name] = value

                # Add day-of-week specific lags
                if segment in hist_data.columns:
                    if f"{segment}_lastweek_sameday" in expected_features:
                        target_same_day = prediction_date - pd.Timedelta(days=7)
                        if target_same_day > last_historical_date:
                            prediction_weekday = prediction_date.weekday()
                            same_weekday_data = hist_data[hist_data["date"].dt.weekday == prediction_weekday]
                            if len(same_weekday_data) > 0:
                                last_same_weekday_idx = same_weekday_data["date"].idxmax()
                                value = hist_data.loc[last_same_weekday_idx, segment]
                                if isinstance(value, pd.Series):
                                    value = value.iloc[0]
                                current_data.loc[:, f"{segment}_lastweek_sameday"] = value
                        else:
                            closest_same_day = hist_data.loc[hist_data["date"] <= target_same_day, "date"].idxmax()
                            value = hist_data.loc[closest_same_day, segment]
                            if isinstance(value, pd.Series):
                                value = value.iloc[0]
                            current_data.loc[:, f"{segment}_lastweek_sameday"] = value

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
                                closest = (
                                    hist_data.loc[hist_data["date"] <= target_date, "date"].idxmax()
                                    if len(hist_data.loc[hist_data["date"] <= target_date]) > 0 else None
                                )
                                if closest is not None and segment in hist_data.columns:
                                    value = hist_data.loc[closest, segment]
                                    if isinstance(value, pd.Series):
                                        value = value.iloc[0]
                                    same_weekday_values.append(value)
                            else:
                                same_weekday_historical = hist_data[hist_data["date"].dt.weekday == prediction_weekday]
                                if len(same_weekday_historical) > 0:
                                    recent_same_weekday = same_weekday_historical.loc[
                                        same_weekday_historical["date"].idxmax(), segment
                                    ]
                                    if isinstance(recent_same_weekday, pd.Series):
                                        recent_same_weekday = recent_same_weekday.iloc[0]
                                    same_weekday_values.append(recent_same_weekday)
                        if same_weekday_values:
                            current_data.loc[:, f"{segment}_avg_4weeks_sameday"] = np.mean(same_weekday_values)
            else:
                # If no historical data available for this lag, use the most recent available data
                if len(hist_data) > 0:
                    last_idx = hist_data["date"].idxmax()
                    for seg in segment_cols:
                        feature_name = f"{seg}_lag_{lag}"
                        if feature_name in expected_features:
                            value = hist_data.loc[last_idx, seg] if seg in hist_data.columns else 0
                            if isinstance(value, pd.Series):
                                value = value.iloc[0]
                            current_data.loc[:, feature_name] = value

        # Calculate rolling features based on available historical data
        windows = [7, 14, 30]
        for window in windows:
            target_window_start = prediction_date - pd.Timedelta(days=window)
            if target_window_start > last_historical_date:
                adjusted_window_start = last_historical_date - pd.Timedelta(days=window)
                adjusted_window_end = last_historical_date
            else:
                adjusted_window_end = min(last_historical_date, prediction_date - pd.Timedelta(days=1))
                adjusted_window_start = adjusted_window_end - pd.Timedelta(days=window)

            window_data = hist_data[
                (hist_data["date"] >= adjusted_window_start) &
                (hist_data["date"] <= adjusted_window_end)
            ]

            if len(window_data) > 0:
                for seg in segment_cols:
                    feature_mean = f"{seg}_rolling_mean_{window}"
                    feature_std = f"{seg}_rolling_std_{window}"

                    if feature_mean in expected_features and seg in window_data.columns:
                        mean_value = float(window_data[seg].mean())
                        current_data.loc[:, feature_mean] = mean_value

                    if feature_std in expected_features and seg in window_data.columns:
                        std_value = float(window_data[seg].std()) if len(window_data) > 1 else 0.0
                        current_data.loc[:, feature_std] = std_value

                if segment in ["recreatief_nl", "recreatief_buitenland"]:
                    rec_cols = [col for col in ["recreatief_nl", "recreatief_buitenland"] if col in window_data.columns]
                    feature_name = f"total_recreational_rolling_{window}"
                    if rec_cols and feature_name in expected_features:
                        value = float(window_data[rec_cols].sum(axis=1).mean())
                        current_data.loc[:, feature_name] = value

                elif segment in ["po", "vo", "student"]:
                    edu_cols = [col for col in ["po", "vo", "student"] if col in window_data.columns]
                    feature_name = f"total_educational_rolling_{window}"
                    if edu_cols and feature_name in expected_features:
                        value = float(window_data[edu_cols].sum(axis=1).mean())
                        current_data.loc[:, feature_name] = value
            else:
                if len(hist_data) > 0:
                    last_values = hist_data.tail(min(window, len(hist_data)))
                    for seg in segment_cols:
                        feature_mean = f"{seg}_rolling_mean_{window}"
                        feature_std = f"{seg}_rolling_std_{window}"

                        if feature_mean in expected_features and seg in last_values.columns:
                            seg_values = last_values[seg]
                            if isinstance(seg_values, pd.DataFrame):
                                seg_values = seg_values.iloc[:, 0]
                            mean_value = seg_values.mean()
                            if isinstance(mean_value, pd.Series):
                                mean_value = mean_value.iloc[0]
                            current_data.loc[:, feature_mean] = float(mean_value)

                        if feature_std in expected_features and seg in last_values.columns:
                            seg_values = last_values[seg]
                            if isinstance(seg_values, pd.DataFrame):
                                seg_values = seg_values.iloc[:, 0]
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
        Create a DataFrame with predictions for all segments, preserving all original rows.
        """
        # Fit the models if not already done
        if not self.models:
            self.fit(df)

        # Prepare results with the full original index
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
                all_predictions[f"{original_name}_pred"].loc[X_train.index] = y_pred_train
                all_predictions[f"{original_name}_actual"].loc[X_train.index] = y_train.values
                all_predictions[f"{original_name}_set"].loc[X_train.index] = "train"

                # Fill in test predictions and labels
                all_predictions[f"{original_name}_pred"].loc[X_test.index] = y_pred_test
                all_predictions[f"{original_name}_actual"].loc[X_test.index] = y_test.values
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
        segments_to_analyze = [segment] if segment else list(self.segment_mappings.values())

        for seg in segments_to_analyze:
            if seg not in self.models:
                continue

            # Get feature importance
            importance = self.models[seg].feature_importances_
            features = self.feature_sets[seg]

            # Create importance DataFrame
            imp_df = pd.DataFrame({
                "feature": features,
                "importance": importance
            }).sort_values("importance", ascending=False)

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
                    data=imp_df.head(top_n),
                    x="importance",
                    y="feature",
                    ax=ax1
                )
                ax1.set_title("Top Individual Features")
                ax1.set_xlabel("Importance Score")

                # Plot feature type importance
                sns.barplot(
                    x=type_importance.values,
                    y=type_importance.index,
                    ax=ax2
                )
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
        path = os.path.join(
            project_root, "Data_Sources", "Data_Modelling", "Predictions", "Segmented_Visitor_Demand_Prediction.csv"
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        predictions_df.to_csv(path, index=False)
        print(f"Predictions saved successfully to: {path}")

    except Exception as e:
        print(f"Error saving file: {e}")
