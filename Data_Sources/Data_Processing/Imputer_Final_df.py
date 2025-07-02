import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Data_Sources.Data_Processing.School_Holidays import create_school_holiday_dataset_openapi
from Data_Sources.Data_Processing.Public_Holiday import create_public_holiday_dataset_openapi
import warnings


warnings.filterwarnings('ignore')

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_cleaned_path = os.path.join(project_root, "Data_Raw", "Data_Cleaned")
datasets = {}

# Load datasets with encoding fallback
for file in os.listdir(path=data_cleaned_path):
    file_name = file.split('.')[0]
    file_path = os.path.join(data_cleaned_path, file)
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    success = False

    for encoding in encodings:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file_path, encoding=encoding)
            elif file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                print(f"Skipping unsupported file format: {file}")
                break
            datasets[file_name] = df
            print(f"Successfully loaded {file_name} with {encoding} encoding")
            success = True
            break
        except UnicodeDecodeError:
            print(f"Failed to decode {file} with {encoding}, next encoding...")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            break
    if not success and file.endswith('.csv'):
        print(f"Could not load {file} with any of the attempted encodings.")

# Pivot hotel table if present
temp_df = datasets.get("hotel_updated")
if temp_df is not None:
    pivot_df = temp_df.pivot(
        index='Datum', columns='Region', values='Value (x1000)'
    )
    pivot_df = pivot_df.reset_index()
    datasets["hotel_updated"] = pivot_df

start_date = "2022-01-26"
end_date = "2025-04-13"

column_rename = {
    "Daily_Sentiment_Summary": {"Publicatiedatum": "Date"},
    "Disruptions_Combined_With_After_Jan2023": {"date": "Date"},
    "forecast_data_cleaned": {"date": "Date"},
    "hotel_updated": {"Datum": "Date"},
    "Amsterdam Events": {"date": "Date", "event_count": "Events_in_Ams"},
    "disruptions_data_historical": {
        "start_time_date": "Date",
        "duration_minutes_sum": "traffic_disruptions_minutes",
        "disruptions_count": "traffic_disruptions_count"
    }
}

needed_columns = {
    "forecast_data_cleaned": [
        "date", "is_open", "school_holiday", "public_holiday", "maat_visitors"
    ],
    "disruptions_data_historical": [
        "start_time_date", "duration_minutes_sum", "disruptions_count"
    ]
}

# Refactor dfs: keep only needed columns and rename for clarity
for file_name, df in datasets.items():
    temp_df = df.copy()
    if file_name in needed_columns:
        needed_col = needed_columns[file_name]
        temp_df = temp_df[needed_col]
    if file_name in column_rename:
        temp_df = temp_df.rename(columns=column_rename[file_name])
    datasets[file_name] = temp_df


def create_merged_dataset(datasets, start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start, end=end, freq='D')
    merged_df = pd.DataFrame({'Date': date_range})
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    for file_name, df in datasets.items():
        print(f"Processing {file_name}...")
        if file_name == "Daily_Sentiment_Summary":
            continue
        temp_df = df.copy()
        if 'Date' not in temp_df.columns:
            print(f"Warning: No 'Date' column in {file_name}, skipping...")
            continue
        temp_df['Date'] = pd.to_datetime(temp_df['Date'])
        if temp_df['Date'].duplicated().any():
            print(
                f"Warning: Found duplicate dates in {file_name}, "
                "keeping first occurrence..."
            )
            temp_df = temp_df.drop_duplicates(subset=['Date'], keep='first')
        cols_to_merge = [col for col in temp_df.columns if col != 'Date']
        if not cols_to_merge:
            print(
                f"Warning: No data columns in {file_name} after removing Date, "
                "skipping..."
            )
            continue
        merged_df = merged_df.merge(
            temp_df[['Date'] + cols_to_merge],
            on='Date',
            how='left'
        )
        print(f"Added {len(cols_to_merge)} columns from {file_name}")
    print(
        f"Final merged dataset has {len(merged_df)} rows and "
        f"{len(merged_df.columns)} columns"
    )
    return merged_df


merged_dataset = create_merged_dataset(datasets, start_date, end_date)


def count_nan_and_zeros(df):
    result = pd.DataFrame(index=df.columns, columns=['NaN_Count', 'Zero_Count'])
    result['NaN_Count'] = df.isna().sum()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            result.loc[column, 'Zero_Count'] = (df[column] == 0).sum()
        elif pd.api.types.is_string_dtype(df[column]):
            result.loc[column, 'Zero_Count'] = (
                (df[column] == '') | (df[column] == '0')
            ).sum()
        else:
            try:
                result.loc[column, 'Zero_Count'] = (df[column] == 0).sum()
            except Exception:
                result.loc[column, 'Zero_Count'] = np.nan
    total_rows = len(df)
    result['NaN_Percentage'] = (result['NaN_Count'] / total_rows * 100).round(2)
    return result


count_nan_and_zeros(merged_dataset)
print(merged_dataset.columns)
merged_dataset = merged_dataset.drop(['school_holiday', 'public_holiday', 'holiday_nl'], axis=1, errors='ignore')


def impute_is_open(df):
    result_df = df.copy()
    result_df['weekday'] = pd.to_datetime(result_df['Date']).dt.dayofweek
    mask_visitors = result_df['Total_Visitors'] > 0
    result_df.loc[mask_visitors, 'is_open'] = 1
    mask_no_visitors = result_df['Total_Visitors'] == 0
    result_df.loc[mask_no_visitors, 'is_open'] = 0
    weekday_pattern = result_df.groupby('weekday')['is_open'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 1
    )
    for weekday, typical_status in weekday_pattern.items():
        mask_weekday = (
            (result_df['weekday'] == weekday) & (result_df['is_open'].isna())
        )
        result_df.loc[mask_weekday, 'is_open'] = typical_status
    return result_df


def impute_school_holiday(df):
    result_df = df.copy()
    school_holiday_df = create_school_holiday_dataset_openapi(start_date, end_date)
    result_df = result_df.merge(
        school_holiday_df[["date", "school_holiday"]],
        left_on="Date", right_on="date", how="left"
    )
    result_df = result_df.drop(columns=["date"])
    return result_df


def impute_public_holiday(df):
    result_df = df.copy()
    public_holiday_df = create_public_holiday_dataset_openapi(start_date, end_date)
    result_df = result_df.merge(
        public_holiday_df[["date", "public_holiday"]],
        left_on="Date", right_on="date", how="left"
    )
    result_df = result_df.drop(columns=["date"])
    return result_df


def impute_all_status(df):
    result_df = df.copy()
    result_df = impute_is_open(result_df)
    result_df = impute_school_holiday(result_df)
    result_df = impute_public_holiday(result_df)
    if 'weekday' in result_df.columns:
        result_df = result_df.drop('weekday', axis=1)
    return result_df


merged_dataset = impute_all_status(merged_dataset)
print("\nMissing values after imputation:")
print(
    merged_dataset[['is_open', 'school_holiday', 'public_holiday']].isna().sum()
)


def impute_disruption_duration(df, column_to_impute):
    result_df = df.copy()
    result_df['weekday'] = pd.to_datetime(result_df['Date']).dt.dayofweek
    features = [
        'MeanTemp_C', 'Precipitation_mm', 'Sunshine_hours', 'weekday',
        'Events_in_Ams', 'public_holiday', 'school_holiday',
        'North Holland (PV)', 'South Holland (PV)', 'Utrecht (PV)'
    ]
    train_mask = ~result_df[column_to_impute].isna()
    X_train = result_df.loc[train_mask, features]
    y_train = result_df.loc[train_mask, column_to_impute]
    X_pred = result_df.loc[~train_mask, features]
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_pred)
    predictions = np.maximum(predictions, 0)
    result_df.loc[~train_mask, column_to_impute] = predictions
    result_df = result_df.drop('weekday', axis=1)
    return result_df


merged_dataset = impute_disruption_duration(merged_dataset, 'traffic_disruptions_minutes')
merged_dataset = impute_disruption_duration(merged_dataset, 'traffic_disruptions_count')

seasonal_path = os.path.join(project_root, "Data_Raw", "Data_Seasonal_Patterns")
datasets_seasonal = {}

for file in os.listdir(path=seasonal_path):
    file_name = file.split('.')[0]
    file_path = os.path.join(seasonal_path, file)
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding)


class AmsterdamTourismFeatureEngine:
    def __init__(self):
        self.monthly_data = {}
        self.feature_mappings = {}

    def load_monthly_data(self, path):
        datasets = {
            'hotels': 'Hotels.csv',
            'theaters': 'Theaters.csv',
            'nemo': 'Nemo.csv',
            'corporate': 'Corporate.csv',
            'museums': 'Museums.csv'
        }
        for file in os.listdir(path=path):
            file_name = file.split('.')[0]
            file_path = os.path.join(path, file)
            key = file_name.lower().replace(' ', '_')
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path, sep=',', encoding=encoding)
                    break
            df_clean = self._clean_monthly_data(df, key)
            self.monthly_data[key] = df_clean

    def _clean_monthly_data(self, df, dataset_type):
        month_cols = [
            'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep',
            'Okt', 'Nov', 'Dec'
        ]
        years = []
        monthly_values = []
        for idx, row in df.iterrows():
            if idx == 0:
                continue
            year_str = str(row.iloc[0])
            if any(char.isdigit() for char in year_str):
                year = int(''.join(filter(str.isdigit, year_str)))
                if 2020 <= year <= 2025:
                    years.append(year)
                    month_vals = []
                    for col in month_cols:
                        if col in df.columns:
                            val = row[col]
                            if pd.notna(val) and val != '':
                                try:
                                    val = float(
                                        str(val).replace(',', '').replace(' ', '')
                                    )
                                    month_vals.append(val)
                                except Exception:
                                    month_vals.append(0)
                        else:
                            month_vals.append(0)
                    monthly_values.append(month_vals)
        df_clean = pd.DataFrame(monthly_values, columns=month_cols, index=years)
        return df_clean

    def create_tourism_features(self, target_dates):
        if isinstance(target_dates[0], str):
            target_dates = pd.to_datetime(target_dates)
        features_df = pd.DataFrame(index=target_dates)
        features_df = self._add_tourism_intensity_features(features_df)
        features_df = self._add_cultural_features(features_df)
        features_df = self._add_seasonal_features(features_df)
        features_df = self._add_trend_features(features_df)
        features_df = self._add_correlation_features(features_df)
        features_df = self._add_daily_distribution_features(features_df)
        return features_df

    def _add_tourism_intensity_features(self, df):
        for date in df.index:
            year, month = date.year, date.month
            month_name = [
                'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                'Sep', 'Okt', 'Nov', 'Dec'
            ][month-1]
            if 'hotels' in self.monthly_data:
                hotel_data = self.monthly_data['hotels']
                if year in hotel_data.index:
                    monthly_hotels = hotel_data.loc[year, month_name]
                    annual_avg = hotel_data.loc[year].mean()
                else:
                    monthly_hotels = hotel_data[month_name].mean()
                    annual_avg = hotel_data.values.mean()
                if annual_avg > 0:
                    df.loc[date, 'hotel_occupancy_index'] = monthly_hotels / annual_avg
                else:
                    df.loc[date, 'hotel_occupancy_index'] = 1
                all_monthly_values = hotel_data[month_name].values
                percentile = (
                    np.sum(all_monthly_values < monthly_hotels) /
                    len(all_monthly_values)
                ) * 100
                df.loc[date, 'tourism_season_strength'] = percentile
            else:
                df.loc[date, 'hotel_occupancy_index'] = 1.0
                df.loc[date, 'tourism_season_strength'] = 50.0
            if 'corporate' in self.monthly_data:
                corp_data = self.monthly_data['corporate']
                if year in corp_data.index and 'hotels' in self.monthly_data:
                    monthly_corp = corp_data.loc[year, month_name]
                    monthly_hotels = self.monthly_data['hotels'].loc[year, month_name]
                    if monthly_hotels > 0:
                        df.loc[date, 'international_visitor_ratio'] = (
                            monthly_corp / monthly_hotels
                        )
                    else:
                        df.loc[date, 'international_visitor_ratio'] = 0
        return df

    def _add_cultural_features(self, df):
        for date in df.index:
            year, month = date.year, date.month
            month_name = [
                'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                'Sep', 'Okt', 'Nov', 'Dec'
            ][month-1]
            cultural_total = 0
            if (
                'theaters' in self.monthly_data and
                year in self.monthly_data['theaters'].index
            ):
                cultural_total += self.monthly_data['theaters'].loc[year, month_name]
            if (
                'museums' in self.monthly_data and
                year in self.monthly_data['museums'].index
            ):
                cultural_total += self.monthly_data['museums'].loc[year, month_name]
            df.loc[date, 'cultural_engagement_score'] = cultural_total
            if (
                'hotels' in self.monthly_data and
                year in self.monthly_data['hotels'].index
            ):
                hotel_visits = self.monthly_data['hotels'].loc[year, month_name]
                if hotel_visits > 0:
                    df.loc[date, 'cultural_vs_tourism_ratio'] = (
                        cultural_total / hotel_visits
                    )
                else:
                    df.loc[date, 'cultural_vs_tourism_ratio'] = 0
            if (
                'nemo' in self.monthly_data and
                'museums' in self.monthly_data and
                year in self.monthly_data['nemo'].index and
                year in self.monthly_data['museums'].index
            ):
                nemo_visits = self.monthly_data['nemo'].loc[year, month_name]
                total_museums = self.monthly_data['museums'].loc[year, month_name]
                if total_museums > 0:
                    df.loc[date, 'nemo_market_share'] = nemo_visits / total_museums
                else:
                    df.loc[date, 'nemo_market_share'] = 0
        return df

    def _add_seasonal_features(self, df):
        if 'hotels' in self.monthly_data:
            hotel_data = self.monthly_data['hotels']
            all_monthly_avgs = {}
            months = [
                'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                'Sep', 'Okt', 'Nov', 'Dec'
            ]
            for i, month_name in enumerate(months):
                month_vals = []
                for year in hotel_data.index:
                    month_vals.append(hotel_data.loc[year, month_name])
                all_monthly_avgs[i+1] = np.mean(month_vals)
            month_rankings = {
                month: rank for rank, month in enumerate(
                    sorted(
                        all_monthly_avgs.keys(),
                        key=lambda x: all_monthly_avgs[x],
                        reverse=True
                    ), 1
                )
            }
            for date in df.index:
                month = date.month
                year = date.year
                month_name = months[month-1]
                df.loc[date, 'month_tourism_rank'] = month_rankings[month]
                if year in hotel_data.index:
                    monthly_val = hotel_data.loc[year, month_name]
                    annual_avg = hotel_data.loc[year].mean()
                    if annual_avg > 0:
                        df.loc[date, 'seasonal_multiplier'] = monthly_val / annual_avg
                    else:
                        df.loc[date, 'seasonal_multiplier'] = 1
                df.loc[date, 'peak_season_flag'] = 1 if month_rankings[month] <= 4 else 0
                df.loc[date, 'shoulder_season_flag'] = (
                    1 if 5 <= month_rankings[month] <= 8 else 0
                )
        return df

    def _add_trend_features(self, df):
        for dataset_name in self.monthly_data.keys():
            data = self.monthly_data[dataset_name]
            for date in df.index:
                year, month = date.year, date.month
                month_name = [
                    'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                    'Sep', 'Okt', 'Nov', 'Dec'
                ][month-1]
                if year in data.index and (year-1) in data.index:
                    current_val = data.loc[year, month_name]
                    prev_year_val = data.loc[year-1, month_name]
                    if prev_year_val > 0:
                        yoy_growth = (
                            (current_val - prev_year_val) / prev_year_val * 100
                        )
                    else:
                        yoy_growth = 0
                    df.loc[date, f'{dataset_name}_yoy_growth'] = yoy_growth
                if year in data.index:
                    months = [
                        'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                        'Sep', 'Okt', 'Nov', 'Dec'
                    ]
                    current_idx = months.index(month_name)
                    if current_idx >= 2:
                        recent_vals = [
                            data.loc[year, months[current_idx-i]] for i in range(3)
                        ]
                        if recent_vals[2] > 0:
                            momentum = (
                                (recent_vals[0] - recent_vals[2]) /
                                recent_vals[2] * 100
                            )
                        else:
                            momentum = 0
                        df.loc[date, f'{dataset_name}_momentum'] = momentum
        return df

    def _add_correlation_features(self, df):
        for date in df.index:
            year, month = date.year, date.month
            month_name = [
                'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                'Sep', 'Okt', 'Nov', 'Dec'
            ][month-1]
            if (
                'corporate' in self.monthly_data and
                'hotels' in self.monthly_data and
                year in self.monthly_data['corporate'].index and
                year in self.monthly_data['hotels'].index
            ):
                corp_val = self.monthly_data['corporate'].loc[year, month_name]
                hotel_val = self.monthly_data['hotels'].loc[year, month_name]
                leisure_proxy = hotel_val - corp_val
                if leisure_proxy > 0:
                    df.loc[date, 'business_leisure_balance'] = corp_val / leisure_proxy
                else:
                    df.loc[date, 'business_leisure_balance'] = 0
            if (
                'museums' in self.monthly_data and
                'nemo' in self.monthly_data and
                year in self.monthly_data['museums'].index and
                year in self.monthly_data['nemo'].index
            ):
                total_museums = self.monthly_data['museums'].loc[year, month_name]
                nemo_visits = self.monthly_data['nemo'].loc[year, month_name]
                other_museums = total_museums - nemo_visits
                if nemo_visits > 0:
                    df.loc[date, 'competitor_activity_index'] = (
                        other_museums / nemo_visits
                    )
                else:
                    df.loc[date, 'competitor_activity_index'] = 0
        return df

    def _add_daily_distribution_features(self, df):
        weekday_factors = [0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.15]
        for date in df.index:
            year, month = date.year, date.month
            month_name = [
                'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                'Sep', 'Okt', 'Nov', 'Dec'
            ][month-1]
            if (
                'nemo' in self.monthly_data and
                year in self.monthly_data['nemo'].index
            ):
                monthly_total = self.monthly_data['nemo'].loc[year, month_name]
                days_in_month = pd.Timestamp(year, month, 1).daysinmonth
                daily_baseline = monthly_total / days_in_month
                weekday_factor = weekday_factors[date.weekday()]
                df.loc[date, 'monthly_expected_baseline'] = (
                    daily_baseline * weekday_factor
                )
            if 'hotel_occupancy_index' in df.columns:
                tourism_pressure = df.loc[date, 'hotel_occupancy_index']
                df.loc[date, 'tourism_pressure_coefficient'] = max(
                    0.5, min(1.5, tourism_pressure)
                )
            if 'competitor_activity_index' in df.columns:
                competition = df.loc[date, 'competitor_activity_index']
                saturation_factor = 1 / (1 + competition * 0.1)
                df.loc[date, 'cultural_saturation_factor'] = saturation_factor
        return df


def create_tourism_features_for_dates(date_range, path):
    engine = AmsterdamTourismFeatureEngine()
    engine.load_monthly_data(path)
    features = engine.create_tourism_features(date_range)
    features = features.fillna(method='ffill').fillna(0)
    return features


date_range = pd.date_range(start_date, end_date, freq='D')
tourism_features = create_tourism_features_for_dates(
    date_range, seasonal_path
).reset_index()
tourism_features = tourism_features.rename(
    columns={'index': 'Date'}, inplace=False
)

big_merged_df = merged_dataset.merge(
    tourism_features,
    on='Date',
    how='left'
)

modelling_path = os.path.join(
    project_root, "Data_Modelling", "Modelling", "Table_for_modelling.csv"
)
os.makedirs(os.path.dirname(modelling_path), exist_ok=True)
big_merged_df.to_csv(modelling_path, index=False)
