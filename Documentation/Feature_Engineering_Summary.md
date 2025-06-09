# Feature Engineering Summary - NEMO Visitor Prediction

## Data Sources

1. **Raw Data Sources**
   - Daily visitor data by segment
   - Weather data
   - Holiday information
   - Amsterdam events data
   - Public transport disruptions
   - Hotel occupancy data
   - Tourism statistics
   - Sentiment data

2. **Date Range**
   - Start: 2022-01-26
   - End: 2025-04-13 (including future predictions)

## Feature Categories

### 1. Temporal Features

**Basic Time Features**
- Day of week (0-6)
- Is weekend (binary)
- Is Monday (binary)
- Is Friday (binary)
- Month (1-12)
- Season (categorical)
  - Winter (Dec-Feb)
  - Spring (Mar-May)
  - Summer (Jun-Aug)
  - Fall (Sep-Nov)

**Advanced Time Features**
- School period indicators
- Holiday periods
- Peak season flags
- Day type (regular, special event, holiday)

### 2. Weather Features

**Raw Weather Data**
- Mean temperature (°C)
- Precipitation (mm)
- Weather conditions

**Engineered Weather Features**
- Good weather flag (Temp > 15°C & Precipitation < 1mm)
- Bad weather flag (Temp < 10°C | Precipitation > 5mm)
- Temperature categories:
  ```
  Cold: < 5°C
  Mild: 5-15°C
  Warm: 15-25°C
  Hot: > 25°C
  ```
- Precipitation categories:
  ```
  Dry: < 0.1mm
  Light: 0.1-5mm
  Heavy: > 5mm
  ```

### 3. Holiday Features

**Basic Holiday Indicators**
- Dutch public holidays
- Dutch school holidays
- International holidays:
  - Germany
  - Belgium
  - France
  - UK
  - Italy

**Derived Holiday Features**
- Total active holidays
- Holiday intensity (multiple overlapping holidays)
- Holiday type categorization
- School break periods
- Holiday season indicators

### 4. Tourism Features

**Hotel and Tourism Data**
- Hotel occupancy rates
- Tourist arrivals
- Regional tourism intensity
- Cultural event attendance

**Derived Tourism Features**
- Tourism intensity score
- Seasonal tourism patterns
- Cultural activity index
- Tourism trend indicators
- Regional distribution features

### 5. Event and Disruption Features

**Event Data**
- Events in Amsterdam count
- Event categories
- Event size/importance

**Transport Disruption Data**
- Disruption count
- Duration of disruptions
- Impact severity
- Alternative transport availability

### 6. Historical Pattern Features

**Lagged Features**
```python
Lag periods:
- Previous day (t-1)
- Previous week (t-7)
- Two weeks ago (t-14)
- Previous month (t-28)
```

**Rolling Statistics**
```python
Windows: [7, 14, 30] days
Metrics:
- Moving average
- Moving median
- Moving standard deviation
- Moving min/max
```

### 7. Interaction Features

**Weather × Holiday Interactions**
- Good weather during holidays
- Bad weather during peak times
- Temperature impact during events

**Event × Tourism Interactions**
- Event impact during high tourism
- Holiday overlap with events
- Weekend × holiday effect

### 8. Segment-Specific Features

**Educational Segments (PO/VO)**
- School period indicators
- Educational event flags
- Grade-specific patterns
- School holiday effects

**Tourist Segments**
- International holiday effects
- Tourism season impact
- Cultural event correlation
- Weather sensitivity

**Local Visitors**
- Local event impact
- Weather sensitivity
- Weekend patterns
- Holiday effects

## Feature Importance

### 1. Visitor Prediction Model

```
Feature Category        Importance
--------------------------------
Temporal               25%
Historical Patterns    20%
Holiday Impact         15%
Weather               15%
Tourism               10%
Events                10%
Disruptions            5%
```

### 2. Crew Size Model

```
Feature Category        Importance
--------------------------------
Visitor Numbers        35%
Temporal               25%
Holiday Impact         15%
Weather               15%
Historical            10%
```

## Feature Engineering Process

1. **Data Cleaning**
   ```mermaid
   graph TD
      A[Raw Data] --> B[Missing Value Imputation]
      B --> C[Outlier Detection]
      C --> D[Data Validation]
      D --> E[Clean Dataset]
   ```

2. **Feature Creation**
   ```mermaid
   graph TD
      A[Clean Data] --> B[Basic Features]
      B --> C[Derived Features]
      C --> D[Interaction Features]
      D --> E[Historical Features]
      E --> F[Final Feature Set]
   ```

3. **Feature Selection**
   - Correlation analysis
   - Feature importance ranking
   - Domain knowledge validation
   - Performance impact testing

## Data Quality Measures

### 1. Missing Value Handling
- Time series interpolation
- Mode/median imputation
- Forward/backward filling
- Domain-specific rules

### 2. Outlier Treatment
- Statistical detection (IQR method)
- Domain knowledge validation
- Contextual outlier handling
- Seasonal pattern preservation

### 3. Feature Validation
- Range checks
- Logical consistency
- Temporal consistency
- Cross-feature validation

## Implementation Details

### 1. Feature Creation Pipeline
```python
def create_feature_set(raw_data):
    # Basic temporal features
    add_temporal_features()
    
    # Weather and holiday features
    add_weather_features()
    add_holiday_features()
    
    # Historical and lagged features
    add_historical_features()
    
    # Interaction features
    create_interaction_features()
    
    # Validation and scaling
    validate_features()
    scale_features()
```

### 2. Feature Update Process
- Daily weather updates
- Weekly tourism updates
- Monthly trend updates
- Real-time event updates

## Usage Guidelines

### 1. Feature Maintenance
- Regular retraining schedule
- Feature importance monitoring
- Data quality checks
- Performance validation

### 2. Feature Selection
- Use domain knowledge
- Consider computational cost
- Balance accuracy vs. complexity
- Monitor feature stability

### 3. Best Practices
- Document feature creation
- Version control features
- Test feature impact
- Monitor feature drift 