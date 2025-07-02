import requests
import pandas as pd
from datetime import datetime, date
from typing import Optional, List, Dict
import time


def fetch_netherlands_subdivisions() -> List[Dict]:
    """
    Fetch Netherlands subdivisions (regions) from OpenHolidaysAPI.
    """
    url = "https://openholidaysapi.org/Subdivisions"
    params = {"countryIsoCode": "NL"}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Could not fetch subdivisions: {e}")
        return []


def fetch_netherlands_school_holidays(start_date: str, end_date: str, subdivision_code: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch school holidays from OpenHolidaysAPI for the Netherlands.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        subdivision_code: Optional subdivision code (e.g., 'NL-NH' for North Holland)
    
    Returns:
        DataFrame with school holiday periods
    """
    url = "https://openholidaysapi.org/SchoolHolidays"
    params = {
        "countryIsoCode": "NL",
        "languageIsoCode": "EN",
        "validFrom": start_date,
        "validTo": end_date
    }
    
    if subdivision_code:
        params["subdivisionCode"] = subdivision_code
    
    print(f"Fetching Dutch school holidays from {start_date} to {end_date}...")
    if subdivision_code:
        print(f"Using subdivision: {subdivision_code}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("No school holiday data found for the specified period")
            return pd.DataFrame()
        
        # Parse the API response
        holidays = []
        for holiday in data:
            holidays.append({
                'id': holiday.get('id', ''),
                'name': holiday.get('name', [{}])[0].get('text', '') if holiday.get('name') else '',
                'startDate': pd.to_datetime(holiday.get('startDate')),
                'endDate': pd.to_datetime(holiday.get('endDate')),
                'subdivisionCode': holiday.get('subdivisions', [{}])[0].get('code', '') if holiday.get('subdivisions') else '',
                'subdivisionName': holiday.get('subdivisions', [{}])[0].get('shortName', '') if holiday.get('subdivisions') else '',
                'type': holiday.get('type', ''),
                'nationwide': holiday.get('nationwide', False)
            })
        
        df = pd.DataFrame(holidays)
        if not df.empty:
            print(f"✅ Successfully fetched {len(df)} school holiday periods")
            if 'subdivisionName' in df.columns:
                regions = df['subdivisionName'].unique()
                print(f"Regions found: {list(regions)}")
        
        return df
        
    except requests.RequestException as e:
        print(f"❌ Error fetching school holidays: {e}")
        return pd.DataFrame()


def create_school_holiday_dataset_openapi(start_date: str = "2022-01-26", 
                                         end_date: str = "2025-04-13", 
                                         subdivision_code: Optional[str] = None) -> pd.DataFrame:
    """
    Create a complete dataset with school holidays using OpenHolidaysAPI.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        subdivision_code: Optional Dutch subdivision code (e.g., 'NL-NH', 'NL-ZH', etc.)
    
    Returns:
        DataFrame with date and school_holiday columns
    """
    
    print(f"Creating school holiday dataset from {start_date} to {end_date}")
    print("Using OpenHolidaysAPI for Netherlands school holidays")
    
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
        
        chunk_holidays = fetch_netherlands_school_holidays(
            chunk_start_str, 
            chunk_end_str, 
            subdivision_code
        )
        
        if not chunk_holidays.empty:
            all_holidays.append(chunk_holidays)
        
        # Move to next chunk
        current_start = chunk_end + pd.DateOffset(days=1)
        
        # Be nice to the API
        time.sleep(0.5)
    
    # Combine all holiday data
    if all_holidays:
        holidays_df = pd.concat(all_holidays, ignore_index=True)
        print(f"\n📊 Total holiday periods found: {len(holidays_df)}")
    else:
        print("⚠️ No holiday data retrieved, creating dataset with no holidays")
        holidays_df = pd.DataFrame()
    
    # Initialize school_holiday column
    df['school_holiday'] = 0
    
    # Apply holiday data
    if not holidays_df.empty:
        print("Applying holiday data to date range...")
        
        for _, holiday in holidays_df.iterrows():
            if pd.notna(holiday['startDate']) and pd.notna(holiday['endDate']):
                mask = (df['date'] >= holiday['startDate']) & (df['date'] <= holiday['endDate'])
                df.loc[mask, 'school_holiday'] = 1
    
    # Add useful metadata columns
    df['day_of_week'] = df['date'].dt.day_name()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.weekday >= 5
    
    return df


def get_available_netherlands_regions() -> None:
    """
    Display available Dutch regions/subdivisions for school holidays.
    """
    print("Fetching available Dutch regions...")
    subdivisions = fetch_netherlands_subdivisions()
    
    if subdivisions:
        print("\n🇳🇱 Available Dutch regions/subdivisions:")
        print("-" * 50)
        for sub in subdivisions:
            code = sub.get('code', 'N/A')
            name = sub.get('shortName', 'N/A')
            official_name = sub.get('officialName', 'N/A')
            print(f"{code}: {name} ({official_name})")
    else:
        print("Could not retrieve subdivision information")
        print("Common Dutch subdivision codes you can try:")
        print("- NL-NH: North Holland (Amsterdam)")
        print("- NL-ZH: South Holland (The Hague, Rotterdam)")
        print("- NL-UT: Utrecht")
        print("- NL-NB: North Brabant")
        print("- Or leave blank for nationwide holidays")


# Example usage and testing
if __name__ == "__main__":
    df = create_school_holiday_dataset_openapi(
        start_date="2022-01-26",
        end_date="2025-04-13",
        subdivision_code=None  # None = nationwide holidays, or specify like "NL-NH"
    )
    print(df[df["school_holiday"] == 1])
