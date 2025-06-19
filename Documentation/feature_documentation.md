# Complete Feature Dictionary

This document contains all unique features across all datasets with their descriptions and calculation formulas where available.

## Core Data Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| Date / date | Date of observation (YYYY-MM-DD) | - |
| Total_Visitors / total_visitors | Total number of visitors on the day | - |
| PO / primary_education | Number of primary education group visitors | - |
| VO / secondary_education | Number of secondary education group visitors | - |
| maat_visitors / crew_size | Crew size category for the day (A, B, C, etc.) | - |
| MeanTemp_C / meantemp_c / avgtemp_c | Mean daily temperature in Celsius | `TG / 10` (from KNMI data) |
| MaxTemp_C | Maximum daily temperature in Celsius | `Original value / 10` |
| MinTemp_C | Minimum daily temperature in Celsius | `Original value / 10` |
| Precipitation_mm / precipitation_mm / totalprecip_mm | Daily precipitation in millimeters | `RH / 10` (from KNMI data) |
| Sunshine_hours | Daily sunshine hours | `SQ / 10` (from KNMI tenths of hours) |
| is_open | Whether the museum was open (1) or closed (0) | `1 if Total_Visitors > 0 else 0 if Total_Visitors == 0 else weekday_mode` |
| school_holiday | Whether it was a school holiday (1) or not (0) | `0 if PO > 0 or VO > 0 else holiday_nl` |
| public_holiday | Whether it was a public holiday (1) or not (0) | `holiday_all if missing; 1 if closed on weekday` |
| holiday_nl | Dutch public holiday flag (1 if Dutch holiday, else 0) | - |
| holiday_all | Any international holiday flag (1 if any country, else 0) | - |
| Events_in_Ams | Number of events in Amsterdam on that day | - |
| duration_minutes | Total duration of disruptions (minutes) | Linear regression imputation on weather, holidays, events, region |
| disruptions_count | Number of disruptions on that day | Linear regression imputation on weather, holidays, events, region |
| North Holland (PV) | Hotel occupancy or tourism index for North Holland province | - |
| South Holland (PV) | Hotel occupancy or tourism index for South Holland province | - |
| Utrecht (PV) | Hotel occupancy or tourism index for Utrecht province | - |

## Visitor Segment Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| Recreatief NL / recreational_domestic | Dutch recreational visitors | - |
| Recreatief Buitenland / recreational_international | International recreational visitors | - |
| Student / students | Student visitors | - |
| Extern / external_events | External/events visitors | - |
| schools | School visitors | - |

## Tourism & Cultural Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| hotel_occupancy_index | Ratio of hotel occupancy for the month to annual average | `monthly_hotel_value / annual_avg_hotel_value` |
| tourism_season_strength | Percentile ranking of tourism season for the month (0-100) | `(number of months with lower value / total months) * 100` |
| international_visitor_ratio | Ratio of corporate (business) stays to hotel stays for the month | `monthly_corporate / monthly_hotels` |
| cultural_engagement_score | Combined score of theater and museum attendance for the month | `theaters + museums` |
| cultural_vs_tourism_ratio | Ratio of cultural attendance to hotel stays | `(theaters + museums) / hotels` |
| nemo_market_share | NEMO's share of total museum attendance for the month | `nemo / museums` |
| month_tourism_rank | Rank of the month by average tourism activity (1=highest) | `rank(monthly_avg)` |
| seasonal_multiplier | Monthly value divided by annual average for seasonality | `monthly_value / annual_avg` |
| peak_season_flag | 1 if month is in top 4 for tourism, else 0 | `1 if month_rank <= 4 else 0` |
| shoulder_season_flag | 1 if month is in middle 4 for tourism, else 0 | `1 if 5 <= month_rank <= 8 else 0` |
| business_leisure_balance | Ratio of business (corporate) to leisure (hotel) stays | `corporate / (hotels - corporate)` |
| competitor_activity_index | Ratio of other museums' attendance to NEMO's attendance | `(museums - nemo) / nemo` |
| monthly_expected_baseline | Expected daily attendance based on monthly totals and weekday | `monthly_nemo / days_in_month * weekday_factor` |
| tourism_pressure_coefficient | Coefficient reflecting tourism pressure (0.5 to 1.5) | `min(1.5, max(0.5, hotel_occupancy_index))` |
| cultural_saturation_factor | Diminishing returns factor for cultural competition | `1 / (1 + competitor_activity_index * 0.1)` |

## Year-over-Year Growth Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| hotels_yoy_growth | Year-over-year growth (%) for hotels | `((current_month - prev_year_month) / prev_year_month) * 100` |
| theaters_yoy_growth | Year-over-year growth (%) for theaters | `((current_month - prev_year_month) / prev_year_month) * 100` |
| nemo_yoy_growth | Year-over-year growth (%) for NEMO | `((current_month - prev_year_month) / prev_year_month) * 100` |
| corporate_yoy_growth | Year-over-year growth (%) for corporate stays | `((current_month - prev_year_month) / prev_year_month) * 100` |
| museums_yoy_growth | Year-over-year growth (%) for museums | `((current_month - prev_year_month) / prev_year_month) * 100` |

## Momentum Features (3-month)

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| hotels_momentum | 3-month momentum (% change) for hotels | `((current_month - month_3_ago) / month_3_ago) * 100` |
| theaters_momentum | 3-month momentum (% change) for theaters | `((current_month - month_3_ago) / month_3_ago) * 100` |
| nemo_momentum | 3-month momentum (% change) for NEMO | `((current_month - month_3_ago) / month_3_ago) * 100` |
| corporate_momentum | 3-month momentum (% change) for corporate stays | `((current_month - month_3_ago) / month_3_ago) * 100` |
| museums_momentum | 3-month momentum (% change) for museums | `((current_month - month_3_ago) / month_3_ago) * 100` |

## Time-based Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| year | Year | `date.dt.year` |
| quarter | Quarter | `date.dt.quarter` |
| month | Month | `date.dt.month` |
| week | Week number | `date.dt.week` |
| weekday / day_of_week | Day of week | `date.dt.dayofweek` |
| day | Day name | `date.dt.day_name()` |
| day_of_year | Day of year | `date.dt.dayofyear` |
| is_month_start | 1 if first day of month, else 0 | `date.dt.is_month_start.astype(int)` |
| is_month_end | 1 if last day of month, else 0 | `date.dt.is_month_end.astype(int)` |
| days_from_today | Days from today | `(pd.Timestamp.now() - date).dt.days` |

## Weather-derived Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| good_weather | 1 if temp > 15°C and precipitation < 1mm, else 0 | `(MeanTemp_C > 15) & (Precipitation_mm < 1)` |
| bad_weather | 1 if temp < 10°C or precipitation > 5mm, else 0 | `(MeanTemp_C < 10) | (Precipitation_mm > 5)` |
| temp_mild | Temperature category: mild | `pd.cut` on meantemp_c |
| temp_warm | Temperature category: warm | `pd.cut` on meantemp_c |
| temp_hot | Temperature category: hot | `pd.cut` on meantemp_c |
| precip_light | Precipitation category: light | `pd.cut` on precipitation_mm |
| precip_heavy | Precipitation category: heavy | `pd.cut` on precipitation_mm |

## Weekend & Holiday Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| is_weekend | 1 if Saturday/Sunday, else 0 | `day_of_week in [5, 6]` |
| is_monday | 1 if Monday, else 0 | `day_of_week == 0` |
| is_friday | 1 if Friday, else 0 | `day_of_week == 4` |
| is_saturday | 1 if Saturday, else 0 | `day_of_week == 5` |
| is_sunday | 1 if Sunday, else 0 | `day_of_week == 6` |
| any_holiday | 1 if any holiday column is 1, else 0 | `max(school_holiday, public_holiday, ...)` |
| total_holidays | Sum of all holiday flags | `sum(holiday columns)` |
| holiday_with_event | 1 if public holiday and event present, else 0 | `(public_holiday == 1) & (event.notna())` |
| school_holiday_weekend | 1 if school holiday and weekend, else 0 | `school_holiday & is_weekend` |
| weekend_holiday | 1 if weekend and holiday, else 0 | `is_weekend * any_holiday` |

## Season Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| season_spring | 1 if spring month, else 0 | Based on month |
| season_summer | 1 if summer month, else 0 | Based on month |
| season_fall | 1 if fall month, else 0 | Based on month |
| season_winter | 1 if winter month, else 0 | Based on month |
| is_summer | 1 if month in [6, 7, 8], else 0 | `month in [6, 7, 8]` |

## Crew Size Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| crew_size_numeric | Numeric encoding of crew size category | `{"Gesloten": 0, "Gesloten maandag": 1, "A min": 2, "A": 3, "B": 4, "C": 5, "D": 6, "Unknown": -1}` |
| high_capacity_day | 1 if predicted visitors > threshold, else 0 | Custom logic |
| low_capacity_day | 1 if predicted visitors < threshold, else 0 | Custom logic |
| weekday_crew_avg | Average crew size for this weekday (historical) | Rolling mean by weekday |

## Visitor Ratio Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| recreatief_nl_ratio / recreational_domestic_percentage | % of Dutch recreational visitors of total | `recreational_domestic / total_visitors * 100` |
| recreatief_buitenland_ratio / recreational_international_percentage | % of international recreational visitors of total | `recreational_international / total_visitors * 100` |
| educational_ratio | Ratio of educational visitors to total predicted | `(PO_pred + VO_pred + Student_pred) / Total Visitors_pred` |
| extern_ratio | Ratio of external visitors to total predicted | `Extern_pred / Total Visitors_pred` |
| schools_percentage | % of school visitors of total | `schools / total_visitors * 100` |
| students_percentage | % of student visitors of total | `students / total_visitors * 100` |
| external_events_percentage | % of external/events visitors of total | `external_events / total_visitors * 100` |

## Lagged Features (Historical Values)

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| [segment]_lag_1 | Value of segment 1 day ago | `segment.shift(1)` |
| [segment]_lag_7 | Value of segment 7 days ago | `segment.shift(7)` |
| [segment]_lag_14 | Value of segment 14 days ago | `segment.shift(14)` |
| [segment]_lag_28 | Value of segment 28 days ago | `segment.shift(28)` |
| crew_size_lag_1 | Crew size category 1 day ago (numeric) | `crew_size_numeric.shift(1)` |
| crew_size_lag_7 | Crew size category 7 days ago (numeric) | `crew_size_numeric.shift(7)` |
| crew_size_lag_14 | Crew size category 14 days ago (numeric) | `crew_size_numeric.shift(14)` |
| crew_size_last_week | Crew size category 7 days ago (numeric) | `crew_size_numeric.shift(7)` |
| [segment]_lastweek_sameday | Value of segment 7 days ago (same weekday) | `segment.shift(7)` |
| [segment]_avg_4weeks_sameday | Mean of segment values on same weekday for last 4 weeks | `(segment.shift(7) + segment.shift(14) + segment.shift(21) + segment.shift(28)) / 4` |

## Rolling Statistics Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| [segment]_rolling_mean_7 | Rolling mean of segment over 7 days | `segment.shift(1).rolling(7).mean()` |
| [segment]_rolling_mean_14 | Rolling mean of segment over 14 days | `segment.shift(1).rolling(14).mean()` |
| [segment]_rolling_mean_30 | Rolling mean of segment over 30 days | `segment.shift(1).rolling(30).mean()` |
| [segment]_rolling_std_7 | Rolling std of segment over 7 days | `segment.shift(1).rolling(7).std()` |
| [segment]_rolling_std_14 | Rolling std of segment over 14 days | `segment.shift(1).rolling(14).std()` |
| [segment]_rolling_std_30 | Rolling std of segment over 30 days | `segment.shift(1).rolling(30).std()` |
| crew_mode_numeric_[window] | Most frequent crew size in last [window] days | `mode(crew_size_numeric.shift(1).rolling(window))` |
| crew_stability_[window] | Fraction of days with most frequent crew size in last [window] days | `mean(crew_size_numeric == mode)` |
| good_weather_freq_[window] | Fraction of days with good weather in last [window] days | `mean(good_weather.shift(1).rolling(window))` |
| holiday_density_[window] | Fraction of days with holiday in last [window] days | `is_holiday.rolling(window).mean()` |

## Aggregated Rolling Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| total_recreational_lag_[N] | Sum of recreational segments N days ago | `recreatief_nl.shift(N) + recreatief_buitenland.shift(N)` |
| total_educational_lag_[N] | Sum of educational segments N days ago | `po.shift(N) + vo.shift(N) + student.shift(N)` |
| total_recreational_rolling_[N] | Rolling mean of recreational segments over window | `(recreatief_nl + recreatief_buitenland).shift(1).rolling(window).mean()` |
| total_educational_rolling_[N] | Rolling mean of educational segments over window | `(po + vo + student).shift(1).rolling(window).mean()` |

## Trend Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| visitor_trend_3d | 3-day trend in predicted visitors | `Total Visitors_pred - Total Visitors_pred.shift(3)` |
| visitor_trend_7d | 7-day trend in predicted visitors | `Total Visitors_pred - Total Visitors_pred.shift(7)` |

## Forecast & Booking Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| forecast_[segment] | Forecast value for visitor segment | - |
| forecast_total | Forecast total visitors | - |
| forecast_total_rounded | Forecast total visitors (rounded) | - |
| forecast_maat | Forecast crew size | - |
| forecast_maat_rounded | Forecast crew size (rounded) | - |
| bookings_primary | Bookings for primary education | - |
| bookings_secondary | Bookings for secondary education | - |
| booking_conversion_primary | Conversion rate for primary education bookings | `primary_education / bookings_primary` |
| booking_conversion_secondary | Conversion rate for secondary education bookings | `secondary_education / bookings_secondary` |
| forecast_accuracy | Ratio of actual to forecast total visitors | `total_visitors / forecast_total` |
| forecast_error | Difference between actual and forecast total visitors | `total_visitors - forecast_total` |
| forecast_error_percentage | Forecast error as percentage of forecast | `forecast_error / forecast_total * 100` |

## Operational Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| nemo_status | 'open' or 'closed' status for NEMO | - |
| opening_time | Opening time | - |
| closing_time | Closing time | - |
| region | Region of school holiday | - |
| event | Event name or description | - |
| forecast_notes | Forecast notes | - |
| forecast_maat_pb | Forecast crew size PB | - |
| forecast_maat_ss | Forecast crew size S&S | - |
| forecast_maat_horeca | Forecast crew size Horeca | - |
| studio_forecast | Studio forecast | - |
| cumulative_forecast | Cumulative forecast | - |
| hours_forecast_pb | Hours forecast PB | - |
| hours_forecast_ss | Hours forecast S&S | - |
| hours_forecast_ss_office | Hours forecast S&S office | - |
| hours_forecast_horeca | Hours forecast Horeca | - |

## Traffic Disruption Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| rdt_station_codes | Station code(s) where disruption occurred | - |
| cause_en | Cause of disruption (in English) | - |
| cause_group | Group/category of the cause | - |
| start_time_date | Date of disruption (YYYY-MM-DD) | `pd.to_datetime(start_time).dt.date` |

## Holiday Effect Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| nl_holiday_effect | Dutch holiday effect for Dutch visitors | `holiday_nl` |
| intl_holiday_effect | International holiday effect for international visitors | `sum(holiday columns except holiday_nl)` |
| edu_holiday_effect | Weighted holiday effect for educational segments | `holiday_nl * 2 + total_holidays` |

## Interaction Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| high_visitors_good_weather | 1 if high capacity and good weather, else 0 | `high_capacity_day * good_weather` |

## Meta Features

| Feature Name | Description | Formula/Calculation |
|--------------|-------------|-------------------|
| source_file | Source CSV file name | - |
| [Segment]_set | Indicates if row was in train or test set | - |

---

**Notes:**
- `[segment]` can be any visitor segment like `Recreatief NL`, `PO`, `VO`, `Student`, `Extern`, etc.
- `[window]` typically refers to time windows like 7, 14, or 30 days
- All ratios return 0 if denominator is 0
- Features marked with `_pred` and `_actual` are used in prediction contexts
- Lagged features use `.shift(N)` to get values from N days ago
- Rolling features exclude the current day using `.shift(1)` before rolling calculation