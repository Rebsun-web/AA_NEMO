import requests
import pandas as pd
import time


def fetch_netherlands_public_holidays(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch public holidays from OpenHolidaysAPI for the Netherlands.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with public holiday dates
    """
    url = "https://openholidaysapi.org/PublicHolidays"
    params = {
        "countryIsoCode": "NL",
        "languageIsoCode": "EN",
        "validFrom": start_date,
        "validTo": end_date
    }
    
    print(f"Fetching Dutch public holidays from {start_date} to {end_date}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("No public holiday data found for the specified period")
            return pd.DataFrame()
        
        # Parse the API response
        holidays = []
        for holiday in data:
            holiday_date = pd.to_datetime(holiday.get('startDate'))
            holiday_name = holiday.get('name', [{}])[0].get('text', '') if holiday.get('name') else ''
            
            holidays.append({
                'date': holiday_date,
                'name': holiday_name,
                'type': holiday.get('type', ''),
                'nationwide': holiday.get('nationwide', False)
            })
        
        df = pd.DataFrame(holidays)
        if not df.empty:
            print(f"✅ Successfully fetched {len(df)} public holidays")
        
        return df
        
    except requests.RequestException as e:
        print(f"❌ Error fetching public holidays: {e}")
        return pd.DataFrame()


def create_public_holiday_dataset_openapi(start_date: str = "2022-01-26",
                                         end_date: str = "2025-04-13") -> pd.DataFrame:
    """
    Create a complete dataset with public holidays using OpenHolidaysAPI.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with date and public_holiday columns
    """
    
    print(f"Creating public holiday dataset from {start_date} to {end_date}")
    print("Using OpenHolidaysAPI for Netherlands public holidays")
    
    # Create full date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'date': date_range})
    
    # The API has a 3-year limit, so we need to fetch data in chunks
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    all_holidays = []
    current_start = start_dt
    
    while current_start < end_dt:
        # Calculate chunk end (max 3 years or actual end date)
        chunk_end = min(current_start + pd.DateOffset(years=3) - pd.DateOffset(days=1), end_dt)
        
        chunk_start_str = current_start.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        
        print(f"Fetching chunk: {chunk_start_str} to {chunk_end_str}")
        
        chunk_holidays = fetch_netherlands_public_holidays(chunk_start_str, chunk_end_str)
        
        if not chunk_holidays.empty:
            all_holidays.append(chunk_holidays)
        
        # Move to next chunk
        current_start = chunk_end + pd.DateOffset(days=1)
        
        # Be nice to the API
        time.sleep(0.5)
    
    # Combine all holiday data
    if all_holidays:
        holidays_df = pd.concat(all_holidays, ignore_index=True)
        print(f"\n📊 Total public holidays found: {len(holidays_df)}")
    else:
        print("⚠️ No public holiday data retrieved")
        holidays_df = pd.DataFrame()
    
    # Initialize public_holiday column
    df['public_holiday'] = 0
    
    # Apply holiday data
    if not holidays_df.empty:
        print("Applying public holiday data to date range...")
        
        # Mark public holidays
        holiday_dates = holidays_df['date'].dt.date
        df['public_holiday'] = df['date'].dt.date.isin(holiday_dates).astype(int)
    
    return df
