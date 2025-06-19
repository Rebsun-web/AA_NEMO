import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import warnings
import sys
import os

# Page configuration
st.set_page_config(
    page_title="NEMO Visitor Prediction Tool", page_icon="🏛️", layout="wide"
)

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Models.Crew_Size_Prediction import CrewSizePredictionModel
    from Models.Segmented_Demand import SegmentedVisitorPredictor
    from Data_Sources.Data_Processing.HolidayChecker import HolidayChecker
    from Data_Sources.Data_Processing.Imputer_Final_df import (
        create_tourism_features_for_dates,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please check that your model classes are in the Models folder")


# Styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .input-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .auto-filled {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
    }
</style>
""",
    unsafe_allow_html=True,
)


def safe_session_state_get(key, default=None):
    """Safely get session state value with default"""
    try:
        return getattr(st.session_state, key, default)
    except (AttributeError, KeyError):
        return default


# Dashboard
class NEMOPredictionDashboard:
    def __init__(self):
        self.holiday_checker = HolidayChecker()

    def load_historical_data(self):
        """Load your historical training data"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(
                project_root,
                "Data_Sources",
                "Data_Modelling",
                "Modelling",
                "Table_for_modelling.csv",
            )
            df = pd.read_csv(file_path)

            # Check for duplicate columns
            if df.columns.duplicated().any():
                st.warning("Duplicate columns found in the data. Removing duplicates.")
                df = df.loc[:, ~df.columns.duplicated()]

            # Check for duplicate dates
            if "Date" in df.columns and df["Date"].duplicated().any():
                st.warning(
                    "Duplicate dates found in historical data. Keeping last occurrence."
                )
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.drop_duplicates(subset=["Date"], keep="last")

            # Store in session state
            st.session_state.historical_data = df

            st.success("Historical data loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Could not load historical data: {str(e)}")
            return False

    def create_input_form(self):
        """Create input form for all required fields"""
        st.markdown(
            '<div class="main-header">🏛️ NEMO Visitor & Crew Prediction</div>',
            unsafe_allow_html=True,
        )

        # Date Selection
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 📅 Date & Basic Info")

        col1, col2, col3 = st.columns(3)

        with col1:
            prediction_date = st.date_input(
                "Prediction Date", value=datetime.now() + timedelta(days=1)
            )

            # Auto-detect day info
            day_name = prediction_date.strftime("%A")
            is_weekend = prediction_date.weekday() >= 5
            st.info(f"📅 **{day_name}** ({'Weekend' if is_weekend else 'Weekday'})")

        with col2:
            is_open = st.selectbox(
                "Museum Open?",
                options=[1, 0],
                format_func=lambda x: "Open" if x else "Closed",
                index=0 if prediction_date.weekday() != 0 else 1,
            )

        with col3:
            school_holiday = st.checkbox("Dutch School Holiday")
            public_holiday = st.checkbox("Dutch Public Holiday")

        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-filled International Holidays
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 🌍 International Holidays (Auto-detected)")

        intl_holidays = self.holiday_checker.check_holidays_for_date(prediction_date)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="auto-filled">', unsafe_allow_html=True)
            st.write(
                f"🇩🇪 German Holiday: {'Yes' if intl_holidays['holiday_de'] else 'No'}"
            )
            st.write(
                f"🇧🇪 Belgian Holiday: {'Yes' if intl_holidays['holiday_be'] else 'No'}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="auto-filled">', unsafe_allow_html=True)
            st.write(
                f"🇫🇷 French Holiday: {'Yes' if intl_holidays['holiday_fr'] else 'No'}"
            )
            st.write(
                f"🇮🇹 Italian Holiday: {'Yes' if intl_holidays['holiday_it'] else 'No'}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="auto-filled">', unsafe_allow_html=True)
            st.write(f"🇬🇧 UK Holiday: {'Yes' if intl_holidays['holiday_gb'] else 'No'}")
            st.write(
                f"🌍 Any International Holiday: {'Yes' if intl_holidays['holiday_all'] else 'No'}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Weather (Exact Numbers)
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 🌤️ Weather Forecast (Exact Values)")

        col1, col2, col3 = st.columns(3)

        with col1:
            mean_temp = st.number_input(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=40.0,
                value=15.0,
                step=0.1,
                help="Enter exact temperature forecast",
            )

        with col2:
            precipitation = st.number_input(
                "Precipitation (mm)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                help="Enter exact precipitation forecast",
            )

        with col3:
            sunshine_hours = st.number_input(
                "Sunshine Hours",
                min_value=0.0,
                max_value=16.0,
                value=6.0,
                step=0.1,
                help="Enter exact sunshine hours forecast",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Transport & Museum Disruptions
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 🚇 Transport & Museum Disruptions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**NS Transport Disruptions:**")
            duration_minutes = st.number_input(
                "Duration (minutes)",
                min_value=0,
                value=0,
                help="Duration of NS disruptions through Amsterdam Central",
            )
            disruptions_count = st.number_input(
                "Lines affected",
                min_value=0,
                value=0,
                help="Number of lines with disruptions through Amsterdam Central",
            )

        with col2:
            st.markdown("**Museum Internal:**")
            total_disruptions = st.number_input(
                "Equipment Disruptions",
                min_value=0,
                max_value=10,
                value=0,
                help="Number of equipment disruptions in the museum",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Operational Categories
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Operational Categories")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            dek_0 = st.number_input("Dek 0", min_value=0, value=0)
            dek_1 = st.number_input("Dek 1", min_value=0, value=0)

        with col2:
            dek_2 = st.number_input("Dek 2", min_value=0, value=0)
            dek_3 = st.number_input("Dek 3", min_value=0, value=0)

        with col3:
            dek_4 = st.number_input("Dek 4", min_value=0, value=0)
            dek_5 = st.number_input("Dek 5", min_value=0, value=0)

        with col4:
            ondersteuning_tb = st.number_input("Ondersteuning TB", min_value=0, value=0)

        st.markdown("</div>", unsafe_allow_html=True)

        # Events and Regional
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 🎪 Events & Regional Visitors")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Events:**")
            events_count = st.number_input(
                "Events in Amsterdam",
                min_value=0,
                value=2,
                help="Number of major events happening in Amsterdam",
            )

        with col2:
            st.markdown("**Regional Visitors (Absolute Numbers):**")
            north_holland = st.number_input(
                "North Holland (PV)",
                min_value=0,
                value=0,
                help="Absolute number of visitors from North Holland",
            )
            south_holland = st.number_input(
                "South Holland (PV)",
                min_value=0,
                value=0,
                help="Absolute number of visitors from South Holland",
            )
            utrecht = st.number_input(
                "Utrecht (PV)",
                min_value=0,
                value=0,
                help="Absolute number of visitors from Utrecht",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Create the input data structure matching your CSV columns exactly
        input_data = {
            "Date": prediction_date,
            "MeanTemp_C": mean_temp,
            "Precipitation_mm": precipitation,
            "Sunshine_hours": sunshine_hours,
            "Dek 0": dek_0,
            "Dek 1": dek_1,
            "Dek 2": dek_2,
            "Dek 3": dek_3,
            "Dek 4": dek_4,
            "Dek 5": dek_5,
            "Ondersteuning TB": ondersteuning_tb,
            "Total_Disruptions": total_disruptions,
            "is_open": is_open,
            "school_holiday": school_holiday,
            "public_holiday": public_holiday,
            "maat_visitors": None,  # Will be predicted by crew model
            "duration_minutes": duration_minutes,
            "disruptions_count": disruptions_count,
            "holiday_nl": public_holiday,
            "holiday_de": intl_holidays["holiday_de"],
            "holiday_be": intl_holidays["holiday_be"],
            "holiday_fr": intl_holidays["holiday_fr"],
            "holiday_it": intl_holidays["holiday_it"],
            "holiday_gb": intl_holidays["holiday_gb"],
            "holiday_all": intl_holidays["holiday_all"],
            "North Holland (PV)": north_holland,
            "South Holland (PV)": south_holland,
            "Utrecht (PV)": utrecht,
            "Events_in_Ams": events_count,
            "Extern": 0,  # Will be predicted
            "PO": 0,  # Will be predicted
            "Recreatief Buitenland": 0,  # Will be predicted
            "Recreatief NL": 0,  # Will be predicted
            "Student": 0,  # Will be predicted
            "VO": 0,  # Will be predicted
            "Total_Visitors": 0,  # Will be predicted
        }

        return input_data

    def get_historical_lagged_features(
        self, input_df, historical_data, prediction_date
    ):
        """Get lagged and rolling features from historical data with fallback to historical patterns"""

        if historical_data is None:
            st.warning("No historical data available for lagged features")
            return input_df

        hist_df = historical_data.copy()
        hist_df["Date"] = pd.to_datetime(hist_df["Date"])

        # Calculate gap
        last_historical_date = hist_df["Date"].max()
        days_gap = (prediction_date - last_historical_date).days

        # Get weekday and month info
        pred_weekday = prediction_date.weekday()
        pred_month = prediction_date.month
        weekday_name = prediction_date.strftime("%A")

        # Create crew_size_numeric if needed
        if (
            "crew_size_numeric" not in hist_df.columns
            and "maat_visitors" in hist_df.columns
        ):
            crew_mapping = {"A min": 2, "A": 3, "B": 4, "C": 5, "D": 6}
            hist_df["crew_size_numeric"] = (
                hist_df["maat_visitors"].map(crew_mapping).fillna(3)
            )

        # Create good_weather column if needed
        if "good_weather" not in hist_df.columns and "MeanTemp_C" in hist_df.columns:
            hist_df["good_weather"] = (
                (hist_df["MeanTemp_C"] > 15) & (hist_df["Precipitation_mm"] < 1)
            ).astype(int)

        # Prepare lagged features dictionary
        lagged_features = {}

        if days_gap > 30:
            st.warning(
                f"⚠️ Large gap ({days_gap} days) between historical data and prediction date. Using historical patterns."
            )

            # Use historical patterns for same weekday and month
            same_weekday_data = hist_df[hist_df["Date"].dt.weekday == pred_weekday]
            same_month_data = hist_df[hist_df["Date"].dt.month == pred_month]
            same_weekday_month_data = hist_df[
                (hist_df["Date"].dt.weekday == pred_weekday)
                & (hist_df["Date"].dt.month == pred_month)
            ]

            # Prioritize same weekday+month, then same weekday, then same month
            if len(same_weekday_month_data) >= 3:
                reference_data = same_weekday_month_data
                st.info(
                    f"Using historical {weekday_name}s from {prediction_date.strftime('%B')}"
                )
            elif len(same_weekday_data) >= 10:
                reference_data = same_weekday_data
                st.info(f"Using historical {weekday_name}s from all months")
            else:
                reference_data = same_month_data
                st.info(f"Using historical data from {prediction_date.strftime('%B')}")

            # Calculate features from reference data
            if len(reference_data) > 0:
                # Crew size features
                if "crew_size_numeric" in reference_data.columns:
                    crew_values = reference_data["crew_size_numeric"].dropna()
                    if len(crew_values) > 0:
                        lagged_features["crew_size_lag_1"] = crew_values.mean()
                        lagged_features["crew_size_lag_7"] = crew_values.mean()
                        lagged_features["crew_size_lag_14"] = crew_values.mean()
                        lagged_features["crew_size_last_week"] = crew_values.mean()
                        lagged_features["crew_mode_numeric_7"] = (
                            crew_values.mode().iloc[0]
                            if len(crew_values.mode()) > 0
                            else crew_values.mean()
                        )
                        lagged_features["crew_mode_numeric_14"] = lagged_features[
                            "crew_mode_numeric_7"
                        ]
                        lagged_features["crew_stability_7"] = 1 - (
                            crew_values.std() / crew_values.mean()
                            if crew_values.mean() > 0
                            else 0
                        )
                        lagged_features["crew_stability_14"] = lagged_features[
                            "crew_stability_7"
                        ]
                        lagged_features["weekday_crew_avg"] = crew_values.mean()

                # Weather features
                if "good_weather" in reference_data.columns:
                    lagged_features["good_weather_freq_7"] = reference_data[
                        "good_weather"
                    ].mean()
                    lagged_features["good_weather_freq_14"] = lagged_features[
                        "good_weather_freq_7"
                    ]

                # Holiday features
                holiday_cols = [
                    col for col in reference_data.columns if "holiday" in col.lower()
                ]
                if holiday_cols:
                    lagged_features["holiday_density_7"] = (
                        reference_data[holiday_cols].any(axis=1).mean()
                    )
                    lagged_features["holiday_density_14"] = lagged_features[
                        "holiday_density_7"
                    ]

        else:
            # Normal processing when gap is reasonable
            # Find the most recent same weekday
            same_weekdays = hist_df[
                hist_df["Date"].dt.weekday == pred_weekday
            ].sort_values("Date")

            if len(same_weekdays) > 0:
                most_recent_same_day = same_weekdays.iloc[-1]
                most_recent_date = most_recent_same_day["Date"]

                st.info(
                    f"📅 Using lagged features from last {weekday_name}: {most_recent_date.strftime('%Y-%m-%d')}"
                )

                # Extract features as before
                if "crew_size_numeric" in most_recent_same_day:
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

                # Calculate rolling features
                hist_index = hist_df[hist_df["Date"] == most_recent_date].index[0]

                # 7-day rolling
                if hist_index >= 6:
                    last_7_days = hist_df.iloc[hist_index - 6 : hist_index + 1]
                    if "crew_size_numeric" in last_7_days.columns:
                        lagged_features["crew_mode_numeric_7"] = (
                            last_7_days["crew_size_numeric"].mode().iloc[0]
                            if len(last_7_days["crew_size_numeric"].mode()) > 0
                            else 3
                        )
                        lagged_features["crew_stability_7"] = (
                            1
                            - (
                                last_7_days["crew_size_numeric"].std()
                                / last_7_days["crew_size_numeric"].mean()
                            )
                            if last_7_days["crew_size_numeric"].mean() > 0
                            else 0.7
                        )

                    if "good_weather" in last_7_days.columns:
                        lagged_features["good_weather_freq_7"] = last_7_days[
                            "good_weather"
                        ].mean()

                    holiday_cols = [
                        col for col in last_7_days.columns if "holiday" in col.lower()
                    ]
                    if holiday_cols:
                        lagged_features["holiday_density_7"] = (
                            last_7_days[holiday_cols].any(axis=1).mean()
                        )

                # 14-day rolling
                if hist_index >= 13:
                    last_14_days = hist_df.iloc[hist_index - 13 : hist_index + 1]
                    if "crew_size_numeric" in last_14_days.columns:
                        lagged_features["crew_mode_numeric_14"] = (
                            last_14_days["crew_size_numeric"].mode().iloc[0]
                            if len(last_14_days["crew_size_numeric"].mode()) > 0
                            else 3
                        )
                        lagged_features["crew_stability_14"] = (
                            1
                            - (
                                last_14_days["crew_size_numeric"].std()
                                / last_14_days["crew_size_numeric"].mean()
                            )
                            if last_14_days["crew_size_numeric"].mean() > 0
                            else 0.6
                        )

                    if "good_weather" in last_14_days.columns:
                        lagged_features["good_weather_freq_14"] = last_14_days[
                            "good_weather"
                        ].mean()

                    if holiday_cols:
                        lagged_features["holiday_density_14"] = (
                            last_14_days[holiday_cols].any(axis=1).mean()
                        )

                # Weekday average
                if len(same_weekdays) > 0:
                    lagged_features["weekday_crew_avg"] = (
                        same_weekdays["crew_size_numeric"].mean()
                        if "crew_size_numeric" in same_weekdays.columns
                        else 3
                    )

        # Set default values for any missing features
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

        # Apply features to input_df
        for feature, default_val in default_values.items():
            if feature in input_df.columns:
                value = lagged_features.get(feature, default_val)
                input_df[feature] = value
                st.success(f"✅ {feature}: {value:.2f}")

        return input_df

    def update_historical_data(self, input_data, visitor_predictions, crew_predictions):
        """Update historical data with new predictions and maintain data quality"""
        try:
            if st.session_state.historical_data is None:
                st.warning("No historical data to update")
                return
            
            # Check if prediction date is within acceptable range
            prediction_date = pd.to_datetime(input_data["Date"])
            today = pd.to_datetime(datetime.now().date())
            days_ahead = (prediction_date - today).days
            
            # Only save predictions for today or tomorrow
            if days_ahead > 1:
                st.info(
                    f"Prediction is {days_ahead} days ahead. Not saving to historical data (only today and tomorrow are saved)."
                )
                return
            
            # Create new row with ALL the input data first
            new_row = input_data.copy()
            
            # Update ONLY the predicted fields with their predictions
            # These are the fields that were predicted, not provided as input
            predicted_segments = [
                "Extern", "PO", "Recreatief Buitenland",
                "Recreatief NL", "Student", "VO"
            ]
            
            # Update visitor segment predictions
            total_visitors = 0
            for segment in predicted_segments:
                if segment in visitor_predictions:
                    values = visitor_predictions[segment]
                    if isinstance(values, dict):
                        # If using confidence predictions, get the point estimate
                        value = float(values.get('point_estimate', 0))
                    elif hasattr(values, "__len__"):
                        value = float(values[0])
                    else:
                        value = float(values)
                    
                    new_row[segment] = value
                    total_visitors += value
            
            # Handle Total Visitors separately - check both with space and underscore
            if "Total Visitors" in visitor_predictions:
                values = visitor_predictions["Total Visitors"]
                if isinstance(values, dict):
                    new_row["Total_Visitors"] = float(values.get('point_estimate', 0))
                elif hasattr(values, "__len__"):
                    new_row["Total_Visitors"] = float(values[0])
                else:
                    new_row["Total_Visitors"] = float(values)
            elif "Total_Visitors" in visitor_predictions:
                values = visitor_predictions["Total_Visitors"]
                if isinstance(values, dict):
                    new_row["Total_Visitors"] = float(values.get('point_estimate', 0))
                elif hasattr(values, "__len__"):
                    new_row["Total_Visitors"] = float(values[0])
                else:
                    new_row["Total_Visitors"] = float(values)
            else:
                # If Total Visitors is not in predictions, calculate it from segments
                new_row["Total_Visitors"] = total_visitors
            
            # Update crew prediction (maat_visitors)
            if crew_predictions is not None and len(crew_predictions) > 0:
                new_row["maat_visitors"] = crew_predictions.iloc[0]["predicted_crew_size"]
            
            # Add metadata to track this is a row with predictions
            new_row["has_predictions"] = True
            new_row["prediction_made_date"] = datetime.now()
            
            # Convert to DataFrame
            new_row_df = pd.DataFrame([new_row])
            
            # Ensure we have all columns from the original table
            original_columns = st.session_state.historical_data.columns.tolist()
            
            # Add any missing columns with default values
            for col in original_columns:
                if col not in new_row_df.columns:
                    # Skip metadata columns that might not exist in original data
                    if col not in ["has_predictions", "prediction_made_date"]:
                        new_row_df[col] = None
            
            # Reorder columns to match historical data (plus any new metadata columns)
            final_columns = original_columns + [col for col in new_row_df.columns if col not in original_columns]
            new_row_df = new_row_df[final_columns]
            
            # Check for duplicate dates
            existing_dates = pd.to_datetime(st.session_state.historical_data["Date"])
            new_date = pd.to_datetime(new_row["Date"])
            
            if new_date in existing_dates.values:
                # OVERWRITE THE ENTIRE ROW - remove old row and add new one
                st.session_state.historical_data = st.session_state.historical_data[
                    existing_dates != new_date
                ]
                
                # Now append the new row
                st.session_state.historical_data = pd.concat(
                    [st.session_state.historical_data, new_row_df],
                    ignore_index=True,
                    sort=False  # Prevent column reordering
                )
                st.info(f"📝 Overwrote existing prediction for {new_date.strftime('%Y-%m-%d')}")
            else:
                # Append new row
                st.session_state.historical_data = pd.concat(
                    [st.session_state.historical_data, new_row_df],
                    ignore_index=True,
                    sort=False  # Prevent column reordering
                )
                st.success(f"📝 Added new row with predictions for {new_date.strftime('%Y-%m-%d')}")
            
            # Sort by date
            st.session_state.historical_data["Date"] = pd.to_datetime(
                st.session_state.historical_data["Date"]
            )
            st.session_state.historical_data = (
                st.session_state.historical_data.sort_values("Date").reset_index(drop=True)
            )
            
            # Save to the original Table_for_modelling.csv location
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            modelling_file = os.path.join(
                project_root,
                "Data_Sources",
                "Data_Modelling",
                "Modelling",
                "Table_for_modelling.csv"
            )
            
            # Create backup before saving
            backup_filename = f'Table_for_modelling_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            backup_dir = os.path.join(
                project_root,
                "Data_Sources",
                "Data_Modelling",
                "Modelling",
                "BackUps"
            )
            
            # Create BackUps directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_file = os.path.join(backup_dir, backup_filename)
            
            if os.path.exists(modelling_file):
                import shutil
                shutil.copy2(modelling_file, backup_file)
                st.info(f"Created backup: {os.path.basename(backup_file)}")
            
            # Save updated data
            st.session_state.historical_data.to_csv(modelling_file, index=False)
            st.success(f"✅ Updated Table_for_modelling.csv with new predictions")
            
            # Clean up old backups (keep only last 5)
            backup_files = sorted([f for f in os.listdir(backup_dir) if f.startswith('Table_for_modelling_backup_')])
            if len(backup_files) > 5:
                for old_backup in backup_files[:-5]:
                    os.remove(os.path.join(backup_dir, old_backup))
            
        except Exception as e:
            st.error(f"Error updating historical data: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    def train_models(self):
        """Train both models"""
        if (
            not hasattr(st.session_state, "historical_data")
            or st.session_state.historical_data is None
        ):
            st.error("Please load historical data first!")
            return False

        try:
            cur_training_model = "visitor prediction models"
            st.info("Training visitor prediction models...")

            # Create a copy without maat_visitors for visitor prediction
            visitor_training_data = st.session_state.historical_data.copy()
            if "maat_visitors" in visitor_training_data.columns:
                visitor_training_data = visitor_training_data.drop(
                    columns=["maat_visitors"]
                )

            visitor_predictor = SegmentedVisitorPredictor()
            visitor_predictor.fit(visitor_training_data)

            # Store in session state
            st.session_state.visitor_predictor = visitor_predictor

            cur_training_model = "crew size prediction model"
            st.info("Training crew size prediction model...")
            crew_predictor = CrewSizePredictionModel()
            crew_predictor.train(st.session_state.historical_data)

            # Store in session state
            st.session_state.crew_predictor = crew_predictor

            st.success("✅ All models trained successfully!")
            return True

        except Exception as e:
            st.error(f"Error training {cur_training_model}: {str(e)}")
            return False

    def make_predictions(self, input_data):
        """Make predictions using trained models with historical lagged features"""
        if (
            not hasattr(st.session_state, "visitor_predictor")
            or not hasattr(st.session_state, "crew_predictor")
            or st.session_state.visitor_predictor is None
            or st.session_state.crew_predictor is None
        ):
            st.error("Models not trained yet!")
            return None, None, None  # Added third None
        
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            prediction_date = pd.to_datetime(input_df["Date"].iloc[0])
            
            # Show gap info
            if st.session_state.historical_data is not None:
                last_historical_date = pd.to_datetime(
                    st.session_state.historical_data["Date"]
                ).max()
                days_gap = (prediction_date - last_historical_date).days
                st.info(
                    f"📊 Historical data: {last_historical_date.strftime('%Y-%m-%d')} → Predicting: {prediction_date.strftime('%Y-%m-%d')} ({days_gap} days gap)"
                )
            
            # 1. Get visitor predictions
            if "maat_visitors" in input_df.columns:
                input_df_for_visitors = input_df.drop(columns=["maat_visitors"])
            else:
                input_df_for_visitors = input_df.copy()
            
            visitor_predictions = st.session_state.visitor_predictor.predict(
                input_df_for_visitors, historical_data=st.session_state.historical_data
            )
            
            # 2. Update input_df with visitor predictions
            input_df_for_crew = input_df.copy()
            for segment, pred in visitor_predictions.items():
                input_df_for_crew[f"{segment}_pred"] = pred
            
            # 3. Add tourism features
            try:
                date_range = [prediction_date]
                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                path = os.path.join(
                    project_root, "Data_Sources", "Data_Raw", "Data_Seasonal_Patterns"
                )
                tourism_features = create_tourism_features_for_dates(date_range, path)
                tourism_features = tourism_features.reset_index()
                tourism_features = tourism_features.rename(columns={"index": "Date"})
                
                # Ensure both dates are datetime and normalized
                input_df_for_crew["Date"] = pd.to_datetime(
                    input_df_for_crew["Date"]
                ).dt.normalize()
                tourism_features["Date"] = pd.to_datetime(
                    tourism_features["Date"]
                ).dt.normalize()
                
                # Merge tourism features
                input_df_for_crew = input_df_for_crew.merge(
                    tourism_features, on="Date", how="left"
                )
            except Exception as e:
                st.warning(f"Could not create tourism features: {e}")
            
            # Set default values for features the crew model expects
            tourism_defaults = {
                "hotel_occupancy_index": 0.5,
                "peak_season_flag": 0,
                "tourism_season_strength": 0.5,
                "cultural_engagement_score": 0.5,
                "cultural_vs_tourism_ratio": 0.5,
                "nemo_market_share": 0.1,
                "month_tourism_rank": 6,
                "shoulder_season_flag": 0,
                "seasonal_multiplier": 1.0,
                "museums_momentum": 0,
                "museums_yoy_growth": 0,
                "theaters_momentum": 0,
                "theaters_yoy_growth": 0,
                "_momentum": 0,
                "_yoy_growth": 0,
                "hotels_momentum": 0,
                "hotels_yoy_growth": 0,
                "nemo_momentum": 0,
                "nemo_yoy_growth": 0,
                "competitor_activity_index": 0.5,
                "tourism_pressure_coefficient": 0.5,
                "cultural_saturation_factor": 0.5,
                "monthly_expected_baseline": 1500,
            }
            
            for col, default_val in tourism_defaults.items():
                if col not in input_df_for_crew.columns:
                    input_df_for_crew[col] = default_val
            
            # 4. Get historical lagged features
            input_df_for_crew = self.get_historical_lagged_features(
                input_df_for_crew, st.session_state.historical_data, prediction_date
            )
            
            # 5. Predict crew size
            crew_predictions = st.session_state.crew_predictor.predict(
                input_df_for_crew
            )
            
            # 6. Create enriched input data dictionary with all features
            enriched_input_data = input_df_for_crew.iloc[0].to_dict()
            
            # Remove the prediction columns that were added for crew model
            pred_columns_to_remove = [col for col in enriched_input_data.keys() if col.endswith('_pred')]
            for col in pred_columns_to_remove:
                enriched_input_data.pop(col, None)
            
            return visitor_predictions, crew_predictions, enriched_input_data
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            with st.expander("📋 Full Error Details"):
                import traceback
                st.code(traceback.format_exc(), language="python")
            return None, None, None

    def make_predictions_with_confidence(self, input_data):
        """Make predictions with confidence intervals and ranges"""
        # Get base predictions
        visitor_predictions_raw, crew_predictions, enriched_input_data = self.make_predictions(input_data)
        
        if visitor_predictions_raw is None:  # Fixed: was checking visitor_predictions
            return None, None
        
        # Store raw predictions for historical update
        self.last_raw_predictions = visitor_predictions_raw
        
        # Calculate is_weekend from the date
        prediction_date = pd.to_datetime(input_data["Date"])
        is_weekend = prediction_date.weekday() >= 5
        
        # Define segment characteristics
        segment_info = {
            "Recreatief NL": {
                "r2": 0.9110,
                "rmse": 286.52,
                "mae": 176.15,
                "confidence_level": "high",
                "historical_std_factor": 0.15,  # 15% variation expected
            },
            "Recreatief Buitenland": {
                "r2": 0.8368,
                "rmse": 214.73,
                "mae": 147.70,
                "confidence_level": "high",
                "historical_std_factor": 0.18,
            },
            "PO": {
                "r2": 0.4933,
                "rmse": 48.59,
                "mae": 33.16,
                "confidence_level": "low",
                "historical_std_factor": 0.35,
            },
            "VO": {
                "r2": 0.3281,
                "rmse": 71.10,
                "mae": 45.97,
                "confidence_level": "low",
                "historical_std_factor": 0.40,
            },
            "Student": {
                "r2": 0.5701,
                "rmse": 32.51,
                "mae": 24.84,
                "confidence_level": "medium",
                "historical_std_factor": 0.25,
            },
            "Extern": {
                "r2": -0.2925,
                "rmse": 84.22,
                "mae": 23.38,
                "confidence_level": "very_low",
                "historical_std_factor": 0.50,
            },
            "Total Visitors": {  # Fixed: was "Total_Visitors"
                "r2": 0.8127,
                "rmse": 559.26,
                "mae": 396.71,
                "confidence_level": "high",
                "historical_std_factor": 0.12,
            },
        }
        
        # Calculate confidence intervals
        confidence_predictions = {}
        
        # Check if we have enough historical data
        has_good_historical_data = True
        if st.session_state.historical_data is not None:
            last_date = pd.to_datetime(st.session_state.historical_data["Date"]).max()
            days_gap = (prediction_date - last_date).days
            has_good_historical_data = days_gap <= 30
        
        # Process each segment from the raw predictions
        for segment, values in visitor_predictions_raw.items():  # Fixed: was visitor_predictions
            if segment in segment_info:
                info = segment_info[segment]
                
                # Get point prediction
                if hasattr(values, "__len__"):
                    point_estimate = float(values[0])
                else:
                    point_estimate = float(values)
                
                # Calculate confidence intervals based on model performance
                if info["confidence_level"] in ["high", "medium"]:
                    # For reliable models, use RMSE for confidence interval
                    if has_good_historical_data:
                        lower_bound = max(0, point_estimate - info["mae"])
                        upper_bound = point_estimate + info["mae"]
                        confidence_range = "Prediction Range (based on MAE)"
                    else:
                        # Wider intervals when historical data is stale
                        lower_bound = max(0, point_estimate - 1.5 * info["rmse"])
                        upper_bound = point_estimate + 1.5 * info["rmse"]
                        confidence_range = "Wider Range (limited recent data)"
                else:
                    # For unreliable models, use historical patterns
                    if (
                        st.session_state.historical_data is not None
                        and segment in st.session_state.historical_data.columns
                    ):
                        # Get historical statistics for similar conditions
                        hist_data = st.session_state.historical_data.copy()
                        hist_data["Date"] = pd.to_datetime(hist_data["Date"])
                        
                        # Filter for similar conditions (weekend vs weekday)
                        if is_weekend:
                            similar_days = hist_data[
                                hist_data["Date"].dt.weekday.isin([5, 6])
                            ]
                        else:
                            similar_days = hist_data[
                                ~hist_data["Date"].dt.weekday.isin([5, 6])
                            ]
                        
                        # Also filter by month if we have enough data
                        month = prediction_date.month
                        month_data = similar_days[
                            similar_days["Date"].dt.month == month
                        ]
                        if len(month_data) >= 5:
                            similar_days = month_data
                        
                        if len(similar_days) > 0 and segment in similar_days.columns:
                            historical_mean = similar_days[segment].mean()
                            historical_std = similar_days[segment].std()
                            
                            # Blend prediction with historical average
                            if info["r2"] < 0:  # Very poor model
                                # Use mostly historical data
                                blended_prediction = (
                                    0.1 * point_estimate + 0.9 * historical_mean
                                )
                            elif info["r2"] < 0.5:  # Poor model
                                blended_prediction = (
                                    0.3 * point_estimate + 0.7 * historical_mean
                                )
                            else:  # Medium confidence
                                blended_prediction = (
                                    0.5 * point_estimate + 0.5 * historical_mean
                                )
                            
                            # Use historical variation for bounds
                            lower_bound = max(
                                0, blended_prediction - 1.5 * historical_std
                            )
                            upper_bound = blended_prediction + 1.5 * historical_std
                            point_estimate = blended_prediction
                            confidence_range = "Historical Pattern Range"
                        else:
                            # Fallback to very wide range
                            lower_bound = max(0, point_estimate * 0.5)
                            upper_bound = point_estimate * 2
                            confidence_range = "High Uncertainty Range"
                    else:
                        # No historical data available
                        lower_bound = max(0, point_estimate * 0.5)
                        upper_bound = point_estimate * 2
                        confidence_range = "High Uncertainty Range"
                
                # Additional check: if point estimate is 0 or very low due to
                # missing features
                if point_estimate < 10 and st.session_state.historical_data is not None:
                    # Try to get a reasonable estimate from historical data
                    try:
                        hist_data = st.session_state.historical_data.copy()
                        hist_data["Date"] = pd.to_datetime(hist_data["Date"])
                        
                        # Get similar days
                        similar_conditions = hist_data[
                            (hist_data["Date"].dt.weekday == prediction_date.weekday())
                            & (hist_data["Date"].dt.month == prediction_date.month)
                        ]
                        
                        if (
                            len(similar_conditions) > 0
                            and segment in similar_conditions.columns
                        ):
                            fallback_estimate = similar_conditions[segment].median()
                            if fallback_estimate > point_estimate:
                                point_estimate = fallback_estimate
                                lower_bound = max(0, fallback_estimate * 0.7)
                                upper_bound = fallback_estimate * 1.3
                                confidence_range = (
                                    "Historical Fallback (zero prediction adjusted)"
                                )
                    except BaseException:
                        pass
                
                confidence_predictions[segment] = {
                    "point_estimate": point_estimate,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "confidence_level": info["confidence_level"],
                    "r2_score": info["r2"],
                    "confidence_range_type": confidence_range,
                    "has_recent_data": has_good_historical_data,
                }

        # Update historical data with raw predictions
        self.update_historical_data(
            enriched_input_data,
            visitor_predictions_raw,  # Use raw predictions, not confidence
            crew_predictions
        )
        
        return confidence_predictions, crew_predictions

    def display_results(self, visitor_predictions, crew_predictions):
        """Display prediction results with confidence intervals"""
        if visitor_predictions is None or crew_predictions is None:
            return

        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")

        # Calculate total visitors from confidence predictions
        total_point = visitor_predictions["Total Visitors"]["point_estimate"]
        total_lower = visitor_predictions["Total Visitors"]["lower_bound"]
        total_upper = visitor_predictions["Total Visitors"]["upper_bound"]

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.metric(
                "📊 Total Expected Visitors",
                f"{int(total_point):,}",
                delta=f"Range: {int(total_lower):,} - {int(total_upper):,}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Crew size prediction
        if len(crew_predictions) > 0:
            crew_size = crew_predictions.iloc[0]["predicted_crew_size"]
            crew_confidence = crew_predictions.iloc[0]["prediction_confidence"]

            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.metric("👥 Predicted Crew Size", crew_size)
                st.markdown("</div>", unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.metric("🎯 Crew Confidence", f"{crew_confidence:.1%}")
                st.markdown("</div>", unsafe_allow_html=True)

        # Detailed segment predictions with confidence
        st.markdown("### 📈 Visitor Segments with Confidence Ranges")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        segments_high_conf = [
            "Recreatief NL",
            "Recreatief Buitenland",
            "Total Visitors",
        ]
        segments_low_conf = ["PO", "VO", "Student", "Extern"]

        with col1:
            st.markdown("#### 🟢 High Confidence Predictions")
            for segment in segments_high_conf:
                if segment in visitor_predictions:
                    pred = visitor_predictions[segment]
                    with st.expander(
                        f"{segment} - {int(pred['point_estimate']):,} visitors"
                    ):
                        st.write(
                            f"**Prediction Range:** {int(pred['lower_bound']):,} - {int(pred['upper_bound']):,}"
                        )
                        st.write(f"**Model R²:** {pred['r2_score']:.3f}")
                        st.write(f"**Range Type:** {pred['confidence_range_type']}")

                        # Show a simple progress bar for the range
                        if pred["upper_bound"] > 0:
                            progress = pred["point_estimate"] / pred["upper_bound"]
                            st.progress(progress)

        with col2:
            st.markdown("#### 🟡 Lower Confidence Predictions")
            for segment in segments_low_conf:
                if segment in visitor_predictions:
                    pred = visitor_predictions[segment]
                    conf_emoji = "🟡" if pred["confidence_level"] == "medium" else "🔴"
                    with st.expander(
                        f"{conf_emoji} {segment} - {int(pred['point_estimate']):,} visitors"
                    ):
                        st.write(
                            f"**Prediction Range:** {int(pred['lower_bound']):,} - {int(pred['upper_bound']):,}"
                        )
                        st.write(f"**Model R²:** {pred['r2_score']:.3f}")
                        st.write(f"**Range Type:** {pred['confidence_range_type']}")
                        if pred["confidence_level"] in ["low", "very_low"]:
                            st.warning(
                                f"⚠️ Low confidence - using {pred['confidence_range_type']}"
                            )

                        # Show a simple progress bar for the range
                        if pred["upper_bound"] > 0:
                            progress = pred["point_estimate"] / pred["upper_bound"]
                            st.progress(progress)

        # Summary statistics
        st.markdown("### 📊 Prediction Summary")

        # Create a summary dataframe
        summary_data = []
        for segment, pred in visitor_predictions.items():
            summary_data.append(
                {
                    "Segment": segment,
                    "Prediction": int(pred["point_estimate"]),
                    "Min": int(pred["lower_bound"]),
                    "Max": int(pred["upper_bound"]),
                    "Confidence": pred["confidence_level"].title(),
                    "R² Score": f"{pred['r2_score']:.3f}",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Data quality warning
        if not all(pred["has_recent_data"] for pred in visitor_predictions.values()):
            st.warning(
                "⚠️ Predictions are based on historical patterns due to lack of recent data. Consider updating your historical dataset for more accurate predictions."
            )


def initialize_session_state():
    """Initialize all session state variables"""
    if "visitor_predictor" not in st.session_state:
        st.session_state.visitor_predictor = None
    if "crew_predictor" not in st.session_state:
        st.session_state.crew_predictor = None
    if "historical_data" not in st.session_state:
        st.session_state.historical_data = None
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False


if __name__ == "__main__":
    initialize_session_state()

    # Initialize dashboard
    dashboard = NEMOPredictionDashboard()

    # Sidebar for model management
    st.sidebar.title("🔧 Model Management")

    if st.sidebar.button("📂 Load Historical Data"):
        dashboard.load_historical_data()
    else:
        print("Failed to load historical data")

    if st.sidebar.button("🎯 Train Models"):
        if st.session_state.historical_data is not None:
            print("Start Training models...")
            dashboard.train_models()
        else:
            st.sidebar.error("Load historical data first!")

    # Model status indicator - check session state safely
    if (
        safe_session_state_get("visitor_predictor") is not None
        and safe_session_state_get("crew_predictor") is not None
    ):
        st.sidebar.success("✅ Models Ready")
    else:
        st.sidebar.warning("⚠️ Models Not Trained")

    # Debug info (optional)
    with st.sidebar.expander("🔍 Debug Info"):
        st.write(
            f"Historical data loaded: {safe_session_state_get('historical_data') is not None}"
        )
        st.write(
            f"Visitor predictor trained: {safe_session_state_get('visitor_predictor') is not None}"
        )
        st.write(
            f"Crew predictor trained: {safe_session_state_get('crew_predictor') is not None}"
        )

    # Main interface
    input_data = dashboard.create_input_form()

    # Show current input summary in expander
    with st.expander("📋 Input Data Summary"):
        display_data = {
            k: v
            for k, v in input_data.items()
            if v not in [0, False, None] or k in ["Date", "is_open"]
        }
        st.json(display_data)

    # Prediction button - check session state safely
    if st.button("🚀 Make Prediction", type="primary", use_container_width=True):
        if (
            safe_session_state_get("visitor_predictor") is None
            or safe_session_state_get("crew_predictor") is None
        ):
            st.error("Please train models first using the sidebar!")
        else:
            with st.spinner("Making predictions..."):
                # Use confidence-weighted predictions
                visitor_pred, crew_pred = dashboard.make_predictions_with_confidence(
                    input_data
                )
                dashboard.display_results(visitor_pred, crew_pred)

    # Help section
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown(
            """
        **Step-by-Step Guide:**

        1. **Load Data**: Click "Load Historical Data" in the sidebar
        2. **Train Models**: Models will auto-train after loading data, or click "Train Models" manually
        3. **Fill Form**: Enter all the required values for your prediction date
        4. **Make Prediction**: Click the prediction button to get results

        **Input Fields:**
        - **Weather**: Enter exact values from weather forecast
        - **Holidays**: Dutch holidays manual, international auto-detected
        - **Transport**: NS disruption data (duration & affected lines)
        - **Museum**: Internal equipment disruptions and operational categories
        - **Events**: Number of major events in Amsterdam
        - **Regional**: Absolute visitor numbers from different provinces

        **Notes:**
        - International holidays are automatically detected using the holidays library
        - All other fields need to be filled manually
        - The tool creates a DataFrame matching your training data structure
        """
        )
