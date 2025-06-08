import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


# Merge DataSets

# In[ ]:
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(
    project_root, "Data_Raw", "Data_Cleaned"
)
datasets = {}

# In[ ]:


for file in os.listdir(path=path):
    file_name = file.split('.')[0]
    file_path = os.path.join(path, file)

    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    success = False
    
    for encoding in encodings:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file_path, encoding=encoding)
            elif file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                # Skip non-CSV/Excel files
                print(f"Skipping unsupported file format: {file}")
                break
                
            datasets[file_name] = df
            print(f"Successfully loaded {file_name} with {encoding} encoding")
            success = True
            break
            
        except UnicodeDecodeError:
            print(f"Failed to decode {file} with {encoding}, trying next encoding...")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            break
    
    if not success and file.endswith('.csv'):
        print(f"Could not load {file} with any of the attempted encodings.")


# In[ ]:


# Pivot Hotel table
temp_df = datasets["hotel_updated"]
pivot_df = temp_df.pivot(index='Datum', columns='Region', values='Value (x1000)')
pivot_df = pivot_df.reset_index()
datasets["hotel_updated"] = pivot_df


# In[ ]:


start_date = "2022-01-26"
end_date = "2025-04-13"

column_rename = {
    "Daily_Sentiment_Summary" : {"Publicatiedatum" : "Date"},
    "Disruptions_Combined_With_After_Jan2023" : {"date" : "Date"},
    "forecast_data_cleaned" : {"date" : "Date"},
    "hotel_updated" : {"Datum" : "Date"},
    "Amsterdam Events" : {"date" : "Date", "event_count" : "Events_in_Ams"},
    "disruptions_data_historical" : {"start_time_date" : "Date", "duration_minutes_sum" : "duration_minutes", "disruptions_count" : "disruptions_count"}
}

needed_columns = {
    "forecast_data_cleaned" : ["date", "is_open", "school_holiday", "public_holiday", "maat_visitors"],
    "disruptions_data_historical" : ["start_time_date", "duration_minutes_sum", "disruptions_count"]
}


# In[ ]:


# Refactor dfs
for file_name, df in datasets.items():
    temp_df = df.copy()
    
    if file_name in needed_columns:
        needed_col = needed_columns[file_name]
        temp_df = temp_df[needed_col]
    
    if file_name in column_rename:
        temp_df = temp_df.rename(columns=column_rename[file_name])
    
    datasets[file_name] = temp_df


# In[ ]:


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
            print(f"Warning: Found duplicate dates in {file_name}, keeping first occurrence...")
            temp_df = temp_df.drop_duplicates(subset=['Date'], keep='first')
        
        cols_to_merge = [col for col in temp_df.columns if col != 'Date']
        
        # Skip if no columns left after removing Date
        if not cols_to_merge:
            print(f"Warning: No data columns in {file_name} after removing Date, skipping...")
            continue
        
        # temp_df = temp_df.rename(columns={col: f"{file_name}_{col}" for col in cols_to_merge})
        
        # Add Date column back
        # cols_to_merge = [f"{file_name}_{col}" for col in cols_to_merge]
        
        # Left merge with our main dataframe
        merged_df = merged_df.merge(
            temp_df[['Date'] + cols_to_merge], 
            on='Date', 
            how='left'
        )
        
        print(f"Added {len(cols_to_merge)} columns from {file_name}")
    
    print(f"Final merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    return merged_df

merged_dataset = create_merged_dataset(datasets, start_date, end_date)


# In[ ]:


def count_nan_and_zeros(df):
    # Create a DataFrame to store the results
    result = pd.DataFrame(index=df.columns, columns=['NaN_Count', 'Zero_Count'])
    
    # Count NaN values in each column
    result['NaN_Count'] = df.isna().sum()
    
    # Count zeros in each column
    for column in df.columns:
        # Need to handle different data types appropriately
        if pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns, count exact zeros
            result.loc[column, 'Zero_Count'] = (df[column] == 0).sum()
        elif pd.api.types.is_string_dtype(df[column]):
            # For string columns, count empty strings and '0' strings
            result.loc[column, 'Zero_Count'] = ((df[column] == '') | (df[column] == '0')).sum()
        else:
            # For other types, try to count zeros but set to NaN if not applicable
            try:
                result.loc[column, 'Zero_Count'] = (df[column] == 0).sum()
            except:
                result.loc[column, 'Zero_Count'] = np.nan
    
    # Calculate percentage of dataset
    total_rows = len(df)
    result['NaN_Percentage'] = (result['NaN_Count'] / total_rows * 100).round(2)
    # result['Zero_Percentage'] = (result['Zero_Count'] / total_rows * 100).round(2)
    
    return result

count_nan_and_zeros(merged_dataset)


# # Handle NaN-values

# ### Statuses imputation

# In[ ]:


def impute_is_open(df):
    result_df = df.copy()
    
    # Add weekday information (0=Monday, 6=Sunday)
    result_df['weekday'] = pd.to_datetime(result_df['Date']).dt.dayofweek
    
    # Rule 1: If there were visitors, the museum was open
    mask_visitors = result_df['Total_Visitors'] > 0
    result_df.loc[mask_visitors, 'is_open'] = 1
    
    # Rule 2: If there were no visitors, the museum was likely closed
    mask_no_visitors = result_df['Total_Visitors'] == 0
    result_df.loc[mask_no_visitors, 'is_open'] = 0
    
    # Rule 3: For remaining NaN values, use typical opening pattern
    weekday_pattern = result_df.groupby('weekday')['is_open'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 1
    )
    
    # Apply weekday pattern to remaining NaN values
    for weekday, typical_status in weekday_pattern.items():
        mask_weekday = (result_df['weekday'] == weekday) & (result_df['is_open'].isna())
        result_df.loc[mask_weekday, 'is_open'] = typical_status
    
    return result_df


def impute_school_holiday(df):
    result_df = df.copy()
    
    # Rule 1: If we have school groups (PO or VO > 0), it's not a school holiday
    mask_school_groups = (result_df['PO'] > 0) | (result_df['VO'] > 0)
    result_df.loc[mask_school_groups, 'school_holiday'] = 0
    
    # Rule 2: Use holiday_nl as reference for remaining NaN values
    result_df.loc[result_df['school_holiday'].isna(), 'school_holiday'] = \
        result_df.loc[result_df['school_holiday'].isna(), 'holiday_nl']
    
    return result_df


def impute_public_holiday(df):
    result_df = df.copy()
    
    # Rule 1: Use holiday_all as base (if any country has a holiday)
    result_df.loc[result_df['public_holiday'].isna(), 'public_holiday'] = \
        result_df.loc[result_df['public_holiday'].isna(), 'holiday_all']
    
    # Rule 2: If museum is closed and it's not a weekend, likely a public holiday
    weekend_mask = result_df['weekday'].isin([5, 6])  # Saturday and Sunday
    closed_weekday_mask = (result_df['is_open'] == 0) & (~weekend_mask)
    result_df.loc[closed_weekday_mask, 'public_holiday'] = 1
    
    return result_df


# Add this to your notebook
def impute_all_status(df):
    result_df = df.copy()
    
    # First impute is_open as it's used by other imputations
    result_df = impute_is_open(result_df)
    
    # Then impute holidays
    result_df = impute_school_holiday(result_df)
    result_df = impute_public_holiday(result_df)
    
    # Drop the temporary weekday column if it exists
    if 'weekday' in result_df.columns:
        result_df = result_df.drop('weekday', axis=1)
    
    return result_df


# In[ ]:


merged_dataset = impute_all_status(merged_dataset)

# Verify results
print("\nMissing values after imputation:")
print(merged_dataset[['is_open', 'school_holiday', 'public_holiday']].isna().sum())


# ### Traffic disruption imputation

# In[ ]:


# Impute duration_minutes using linear regression
def impute_disruption_duration(df, column_to_impute):
    result_df = df.copy()
    
    # Add weekday feature
    result_df['weekday'] = pd.to_datetime(result_df['Date']).dt.dayofweek
    
    # Select features for the model
    features = [
        'MeanTemp_C', 
        'Precipitation_mm', 
        'Sunshine_hours', 
        'weekday',
        'Events_in_Ams',
        'public_holiday',
        'school_holiday',
        'North Holland (PV)',
        'South Holland (PV)',
        'Utrecht (PV)'
    ]
    
    # Split into training and prediction sets
    train_mask = ~result_df[column_to_impute].isna()
    X_train = result_df.loc[train_mask, features]
    y_train = result_df.loc[train_mask, column_to_impute]
    X_pred = result_df.loc[~train_mask, features]
    
    # Train model and make predictions
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_pred)
    
    # Ensure non-negative durations
    predictions = np.maximum(predictions, 0)
    
    # Fill in predictions
    result_df.loc[~train_mask, column_to_impute] = predictions
    
    # Clean up temporary columns
    result_df = result_df.drop('weekday', axis=1)
    
    return result_df

# Apply the imputation
merged_dataset = impute_disruption_duration(merged_dataset, 'duration_minutes')
merged_dataset = impute_disruption_duration(merged_dataset, 'disruptions_count')


# ### Add Seasonal Features

# In[ ]:


from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# In[ ]:
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(
    project_root, "Data_Raw", "Data_Seasonal_Patterns"
)

datasets = {}


# In[ ]:


for file in os.listdir(path=path):
    file_name = file.split('.')[0]
    file_path = os.path.join(path, file)

    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding)


# In[ ]:


class AmsterdamTourismFeatureEngine:
    def __init__(self):
        self.monthly_data = {}
        self.feature_mappings = {}
        
    def load_monthly_data(self, path):
        """Load all monthly CSV files"""

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
                    df = pd.read_csv(file_path,sep=',', encoding=encoding)
                    break

            df_clean = self._clean_monthly_data(df, key)
            self.monthly_data[key] = df_clean


    def _clean_monthly_data(self, df, dataset_type):
        """Clean and standardize monthly data format"""
        # Handle different column structures
        month_cols = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
        
        # Extract years from first column or infer from row count
        years = []
        monthly_values = []
        
        for idx, row in df.iterrows():
            if idx == 0:  # Skip header if needed
                continue
            
            year_str = str(row.iloc[0])
            if any(char.isdigit() for char in year_str):
                year = int(''.join(filter(str.isdigit, year_str)))
                if 2020 <= year <= 2025:
                    years.append(year)
                    
                    # Extract monthly values
                    month_vals = []
                    for col in month_cols:
                        if col in df.columns:
                            val = row[col]
                            # Clean numeric values
                            if pd.notna(val) and val != '':
                                try:
                                    val = float(str(val).replace(',', '').replace(' ', ''))
                                    month_vals.append(val)
                                except:
                                    month_vals.append(0)
                            else:
                                month_vals.append(0)
                    monthly_values.append(month_vals)
        
        # Create clean DataFrame
        df_clean = pd.DataFrame(monthly_values, columns=month_cols, index=years)
        return df_clean

    
    def create_tourism_features(self, target_dates):
        """Create all tourism-based features for given dates"""
        # Convert dates to datetime if they're strings
        if isinstance(target_dates[0], str):
            target_dates = pd.to_datetime(target_dates)
            
        features_df = pd.DataFrame(index=target_dates)
        
        # 1. Tourism Intensity Indicators
        features_df = self._add_tourism_intensity_features(features_df)
        
        # 2. Cultural Activity Features  
        features_df = self._add_cultural_features(features_df)
        
        # 3. Monthly Pattern Features
        features_df = self._add_seasonal_features(features_df)
        
        # 4. Trend & Growth Features
        features_df = self._add_trend_features(features_df)
        
        # 5. Cross-Sector Correlation Features
        features_df = self._add_correlation_features(features_df)
        
        # 6. Day-Level Application Features
        features_df = self._add_daily_distribution_features(features_df)
        
        return features_df
    

    def _add_tourism_intensity_features(self, df):
        """Tourism intensity indicators"""
        for date in df.index:
            year, month = date.year, date.month
            month_name = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'][month-1]
            
            # Hotel occupancy index (normalized by annual average)
            if 'hotels' in self.monthly_data:
                hotel_data = self.monthly_data['hotels']
                
                if year in hotel_data.index:
                    # Use exact year data if available
                    monthly_hotels = hotel_data.loc[year, month_name]
                    annual_avg = hotel_data.loc[year].mean()
                else:
                    # Use average across all years for this month
                    monthly_hotels = hotel_data[month_name].mean()
                    # For annual average, use the overall mean across all years/months
                    annual_avg = hotel_data.values.mean()
                    # print(f"Using average {month_name} data: {monthly_hotels:.2f} (overall avg: {annual_avg:.2f})")
                
                df.loc[date, 'hotel_occupancy_index'] = monthly_hotels / annual_avg if annual_avg > 0 else 1
                
                # Tourism season strength (percentile ranking using multi-year data)
                all_monthly_values = hotel_data[month_name].values  # All years for this month
                percentile = (np.sum(all_monthly_values < monthly_hotels) / len(all_monthly_values)) * 100
                df.loc[date, 'tourism_season_strength'] = percentile
            else:
                # No hotel data available - use defaults
                df.loc[date, 'hotel_occupancy_index'] = 1.0
                df.loc[date, 'tourism_season_strength'] = 50.0
                    
            # International visitor ratio (corporate meetings proxy)
            if 'corporate' in self.monthly_data:
                corp_data = self.monthly_data['corporate']
                if year in corp_data.index and 'hotels' in self.monthly_data:
                    monthly_corp = corp_data.loc[year, month_name]
                    monthly_hotels = self.monthly_data['hotels'].loc[year, month_name]
                    df.loc[date, 'international_visitor_ratio'] = (
                        monthly_corp / monthly_hotels if monthly_hotels > 0 else 0
                    )
        
        return df
    

    def _add_cultural_features(self, df):
        """Cultural activity features"""
        for date in df.index:
            year, month = date.year, date.month
            month_name = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'][month-1]
            
            # Cultural engagement score
            cultural_total = 0
            if 'theaters' in self.monthly_data and year in self.monthly_data['theaters'].index:
                cultural_total += self.monthly_data['theaters'].loc[year, month_name]
            if 'museums' in self.monthly_data and year in self.monthly_data['museums'].index:
                cultural_total += self.monthly_data['museums'].loc[year, month_name]
                
            df.loc[date, 'cultural_engagement_score'] = cultural_total
            
            # Cultural vs tourism ratio
            if 'hotels' in self.monthly_data and year in self.monthly_data['hotels'].index:
                hotel_visits = self.monthly_data['hotels'].loc[year, month_name]
                df.loc[date, 'cultural_vs_tourism_ratio'] = (
                    cultural_total / hotel_visits if hotel_visits > 0 else 0
                )
            
            # NEMO market share
            if ('nemo' in self.monthly_data and 'museums' in self.monthly_data and 
                year in self.monthly_data['nemo'].index and 
                year in self.monthly_data['museums'].index):
                nemo_visits = self.monthly_data['nemo'].loc[year, month_name]
                total_museums = self.monthly_data['museums'].loc[year, month_name]
                df.loc[date, 'nemo_market_share'] = (
                    nemo_visits / total_museums if total_museums > 0 else 0
                )
        
        return df
    

    def _add_seasonal_features(self, df):
        """Monthly pattern features"""
        # Calculate overall tourism activity for ranking
        if 'hotels' in self.monthly_data:
            hotel_data = self.monthly_data['hotels']
            
            # Monthly rankings across all years
            all_monthly_avgs = {}
            months = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
            
            for i, month_name in enumerate(months):
                month_vals = []
                for year in hotel_data.index:
                    month_vals.append(hotel_data.loc[year, month_name])
                all_monthly_avgs[i+1] = np.mean(month_vals)
            
            # Rank months by average activity
            month_rankings = {month: rank for rank, month in enumerate(
                sorted(all_monthly_avgs.keys(), key=lambda x: all_monthly_avgs[x], reverse=True), 1
            )}
            
            for date in df.index:
                month = date.month
                year = date.year
                month_name = months[month-1]
                
                # Month tourism rank (1 = highest tourism month)
                df.loc[date, 'month_tourism_rank'] = month_rankings[month]
                
                # Seasonal multiplier
                if year in hotel_data.index:
                    monthly_val = hotel_data.loc[year, month_name]
                    annual_avg = hotel_data.loc[year].mean()
                    df.loc[date, 'seasonal_multiplier'] = monthly_val / annual_avg if annual_avg > 0 else 1
                
                # Peak season flag (top 4 months)
                df.loc[date, 'peak_season_flag'] = 1 if month_rankings[month] <= 4 else 0
                
                # Shoulder season flag (middle 4 months)
                df.loc[date, 'shoulder_season_flag'] = 1 if 5 <= month_rankings[month] <= 8 else 0
        
        return df
    

    def _add_trend_features(self, df):
        """Trend and growth features"""
        for dataset_name in self.monthly_data.keys():
            data = self.monthly_data[dataset_name]
            
            for date in df.index:
                year, month = date.year, date.month
                month_name = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'][month-1]
                
                # Year-over-year growth
                if year in data.index and (year-1) in data.index:
                    current_val = data.loc[year, month_name]
                    prev_year_val = data.loc[year-1, month_name]
                    yoy_growth = ((current_val - prev_year_val) / prev_year_val * 100 if prev_year_val > 0 else 0)
                    df.loc[date, f'{dataset_name}_yoy_growth'] = yoy_growth
                
                # 3-month momentum (if we have previous months)
                if year in data.index:
                    months = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
                    current_idx = months.index(month_name)
                    
                    if current_idx >= 2:  # Need at least 3 months
                        recent_vals = [data.loc[year, months[current_idx-i]] for i in range(3)]
                        momentum = (recent_vals[0] - recent_vals[2]) / recent_vals[2] * 100 if recent_vals[2] > 0 else 0
                        df.loc[date, f'{dataset_name}_momentum'] = momentum
        
        return df
    
    def _add_correlation_features(self, df):
        """Cross-sector correlation features"""
        for date in df.index:
            year, month = date.year, date.month
            month_name = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'][month-1]
            
            # Business vs leisure balance
            if ('corporate' in self.monthly_data and 'hotels' in self.monthly_data and
                year in self.monthly_data['corporate'].index and 
                year in self.monthly_data['hotels'].index):
                
                corp_val = self.monthly_data['corporate'].loc[year, month_name]
                hotel_val = self.monthly_data['hotels'].loc[year, month_name]
                leisure_proxy = hotel_val - corp_val  # Rough estimate
                
                df.loc[date, 'business_leisure_balance'] = (
                    corp_val / leisure_proxy if leisure_proxy > 0 else 0
                )
            
            # Competitor activity index (other museums vs NEMO)
            if ('museums' in self.monthly_data and 'nemo' in self.monthly_data and
                year in self.monthly_data['museums'].index and 
                year in self.monthly_data['nemo'].index):
                
                total_museums = self.monthly_data['museums'].loc[year, month_name]
                nemo_visits = self.monthly_data['nemo'].loc[year, month_name]
                other_museums = total_museums - nemo_visits
                
                df.loc[date, 'competitor_activity_index'] = (
                    other_museums / nemo_visits if nemo_visits > 0 else 0
                )
        
        return df
    
    def _add_daily_distribution_features(self, df):
        """Day-level application features"""
        # Standard daily distribution patterns (weekday/weekend effects)
        weekend_boost = 1.2  # Museums typically see 20% boost on weekends
        weekday_factors = [0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.15]  # Mon-Sun
        
        for date in df.index:
            year, month = date.year, date.month
            month_name = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'][month-1]
            
            # Monthly expected baseline (distributed daily)
            if 'nemo' in self.monthly_data and year in self.monthly_data['nemo'].index:
                monthly_total = self.monthly_data['nemo'].loc[year, month_name]
                days_in_month = pd.Timestamp(year, month, 1).daysinmonth
                
                # Simple daily distribution
                daily_baseline = monthly_total / days_in_month
                
                # Apply weekday factor
                weekday_factor = weekday_factors[date.weekday()]
                df.loc[date, 'monthly_expected_baseline'] = daily_baseline * weekday_factor
            
            # Tourism pressure coefficient
            if 'hotel_occupancy_index' in df.columns:
                tourism_pressure = df.loc[date, 'hotel_occupancy_index']
                # Higher tourism = more competition for attention
                df.loc[date, 'tourism_pressure_coefficient'] = max(0.5, min(1.5, tourism_pressure))
            
            # Cultural saturation factor
            if 'competitor_activity_index' in df.columns:
                competition = df.loc[date, 'competitor_activity_index']
                # Higher competition = lower relative appeal
                saturation_factor = 1 / (1 + competition * 0.1)  # Diminishing returns
                df.loc[date, 'cultural_saturation_factor'] = saturation_factor
        
        return df

# Usage example:
def create_tourism_features_for_dates(date_range, path):
    """
    Main function to create features for a date range
    
    Parameters:
    date_range: list of dates or pandas date range
    data_file_paths: dict mapping dataset names to file paths
    
    Returns:
    DataFrame with tourism features
    """
    
    # Initialize feature engine
    engine = AmsterdamTourismFeatureEngine()
    
    # Load data
    engine.load_monthly_data(path)
    
    # Create features
    features = engine.create_tourism_features(date_range)
    
    # Fill any remaining NaN values
    features = features.fillna(method='ffill').fillna(0)
    
    return features


# In[ ]:


date_range = pd.date_range(start_date, end_date, freq='D')
tourism_features = create_tourism_features_for_dates(date_range, path).reset_index()


# In[ ]:


tourism_features = tourism_features.rename(columns={'index': 'Date'}, inplace=False)


# # Merge

# In[ ]:


big_merged_df = merged_dataset.merge(
            tourism_features, 
            on='Date', 
            how='left'
        )


# In[ ]:
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modelling_path = os.path.join(
    project_root, "Data_Cleaned", "Modelling", "Table_for_modelling.csv"
)

big_merged_df.to_csv(modelling_path, index=False)
