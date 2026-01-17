# CAISO Resource Adequacy Price Forecast

A comprehensive forecasting tool for California ISO (CAISO) Resource Adequacy (RA) prices for years 2026-2030.

## Overview

This module provides multi-year RA price forecasts across five CAISO zones, incorporating key market drivers including load growth, renewable energy penetration, generator retirements, and battery storage deployment.

## Features

- **Multi-Year Forecasting**: Generates forecasts for 2026-2030
- **Multiple Zones**: Covers all major CAISO local capacity areas:
  - System (system-wide)
  - Big Creek/Ventura
  - LA Basin
  - San Diego/Imperial Valley
  - Sierra
- **Monthly Granularity**: Captures seasonal patterns in RA pricing
- **Confidence Intervals**: Provides 95% confidence bounds for uncertainty quantification
- **Comprehensive Visualizations**: Generates 5 types of plots for analysis
- **Historical Context**: Includes synthetic historical data (2023-2025) for trend analysis

## Quick Start

### Basic Usage

Run the main forecast script:

```bash
python caiso_ra_forecast.py
```

This will generate:
- Forecast CSV files in `forecasts/`
- Visualization plots in `plots/`
- Metadata JSON in `forecasts/`

### Programmatic Usage

```python
from src.caiso_ra_forecaster import CAISORAForecaster
from src.caiso_ra_visualization import CAISORAVisualizer

# Initialize forecaster
forecaster = CAISORAForecaster()

# Generate forecast
forecast_data = forecaster.generate_forecast()

# Get annual summary
summary = forecaster.get_annual_summary()

# Save results
forecaster.save_forecast()
forecaster.save_metadata()

# Create visualizations
historical = forecaster.generate_historical_baseline()
visualizer = CAISORAVisualizer(forecast_data, historical)
visualizer.generate_all_plots()
```

### Example Scripts

Run the comprehensive examples:

```bash
python example_caiso_ra_usage.py
```

This demonstrates:
1. Basic forecast generation
2. Custom configuration
3. Zone-specific analysis
4. Visualization generation
5. Monthly analysis
6. Data export
7. Cost calculations

## Methodology

### Forecasting Approach

The forecast uses a **trend-based projection model** with compound growth factors:

#### Growth Factors

| Factor | Rate | Impact |
|--------|------|--------|
| Annual Load Growth | +1.5% | Increasing demand for capacity |
| Renewable Penetration | +2.5% | Need for firming capacity |
| Generator Retirements | +2.0% | Reduced supply (gas plant retirements) |
| Battery Storage | -1.5% | New storage reducing RA prices |
| **Net Growth Rate** | **+4.5%** | **Compound annual price growth** |

#### Price Calculation

For each zone and month, the forecast price is:

```
Forecasted Price = Base Price × (1 + Net Growth Rate)^(Years Ahead)
```

Where:
- **Base Price**: 2025 reference prices by zone and month
- **Years Ahead**: Number of years from base year (2025)
- **Net Growth Rate**: 4.5% (sum of all growth factors)

#### Uncertainty Quantification

- **Base Volatility**: 10% standard deviation
- **Volatility Growth**: Increases 2% per year (reflects greater uncertainty in distant future)
- **Confidence Intervals**: 95% (±1.96 standard deviations)

### Base Prices (2025)

Average monthly RA prices used as baseline ($/kW-month):

| Zone | Summer Peak (Jul-Aug) | Winter Off-Peak (Jan-Mar) |
|------|-----------------------|---------------------------|
| System | $8.50-9.25 | $4.00-4.50 |
| Big Creek/Ventura | $12.00-13.50 | $5.75-6.50 |
| LA Basin | $13.50-15.00 | $6.25-7.00 |
| San Diego/Imperial Valley | $11.00-12.00 | $5.25-5.75 |
| Sierra | $8.00-8.75 | $3.75-4.25 |

## Output Files

### Forecast Data CSV

Format: `forecasts/caiso_ra_forecast_[timestamp].csv`

Columns:
- `year`: Forecast year (2026-2030)
- `month`: Month name
- `month_num`: Month number (1-12)
- `zone`: CAISO zone
- `ra_price`: Forecasted RA price ($/kW-month)
- `lower_bound`: 95% CI lower bound
- `upper_bound`: 95% CI upper bound
- `growth_factor`: Applied growth multiplier
- `date`: Timestamp for the month

### Annual Summary CSV

Format: `forecasts/caiso_ra_annual_summary_[timestamp].csv`

Columns:
- `zone`: CAISO zone
- `year`: Forecast year
- `avg_price`: Average annual price
- `min_price`: Minimum monthly price
- `max_price`: Maximum monthly price
- `std_dev`: Standard deviation
- `avg_lower_bound`: Average lower CI bound
- `avg_upper_bound`: Average upper CI bound

### Metadata JSON

Format: `forecasts/caiso_ra_metadata_[timestamp].json`

Contains:
- Forecast type and generation timestamp
- Forecast years and zones
- Methodology description
- Growth factors and assumptions
- Units and confidence interval level

## Visualizations

The tool generates five comprehensive visualizations:

### 1. Multi-Year Trends
**File**: `plots/caiso_ra_multi_year_trends_[timestamp].png`

Shows price trajectories for all zones across 2026-2030 with:
- Historical baseline (2023-2025)
- Forecast prices
- 95% confidence intervals
- Year separators

### 2. Zone Comparison
**File**: `plots/caiso_ra_zone_comparison_[timestamp].png`

Bar chart comparing average annual prices across zones and years.

### 3. Seasonal Patterns
**File**: `plots/caiso_ra_seasonal_patterns_[timestamp].png`

Monthly price patterns for each forecast year, showing:
- Summer peaks (June-September)
- Winter lows (December-March)
- Shoulder months (April-May, October-November)

### 4. Price Heatmap
**File**: `plots/caiso_ra_price_heatmap_[timestamp].png`

Heatmap of average annual prices by zone and year.

### 5. Growth Trends
**File**: `plots/caiso_ra_growth_trends_[timestamp].png`

Two-panel plot showing:
- Average annual prices over time
- Year-over-year growth rates by zone

## Customization

### Custom Configuration

Create a custom configuration dictionary:

```python
custom_config = {
    'base_year': 2025,
    'forecast_years': [2026, 2027, 2028, 2029, 2030],

    # Adjust growth factors
    'annual_load_growth': 0.02,  # 2% load growth
    'renewable_penetration_impact': 0.03,  # 3% renewable impact
    'retirement_impact': 0.025,  # 2.5% retirement impact
    'battery_storage_impact': -0.02,  # -2% battery impact

    # Modify uncertainty
    'base_volatility': 0.12,  # 12% base volatility
    'volatility_growth': 0.025,  # 2.5% volatility growth

    # Select specific zones
    'zones': ['System', 'LA Basin']
}

forecaster = CAISORAForecaster(config=custom_config)
```

### Zone-Specific Analysis

```python
# Analyze specific zone
la_basin_data = forecast_data[forecast_data['zone'] == 'LA Basin']

# Calculate summer average
summer_months = [6, 7, 8, 9]
summer_avg = la_basin_data[
    la_basin_data['month_num'].isin(summer_months)
]['ra_price'].mean()
```

### Cost Estimation

Calculate procurement costs for a given capacity:

```python
capacity_mw = 500  # 500 MW capacity requirement
zone = 'LA Basin'
year = 2026

zone_year_data = forecast_data[
    (forecast_data['zone'] == zone) &
    (forecast_data['year'] == year)
]

# Calculate annual cost
annual_cost = sum(
    row['ra_price'] * capacity_mw * 1000  # Convert MW to kW
    for _, row in zone_year_data.iterrows()
)

print(f"Annual RA cost: ${annual_cost:,.2f}")
```

## Forecast Results Summary

### System-Wide Average Prices

| Year | System Avg ($/kW-month) | YoY Change |
|------|-------------------------|------------|
| 2026 | $7.86 | - |
| 2027 | $8.21 | +4.5% |
| 2028 | $8.58 | +4.5% |
| 2029 | $8.97 | +4.5% |
| 2030 | $9.37 | +4.5% |

### Key Insights

1. **Consistent Growth**: All zones show steady 4.5% annual price growth
2. **LA Basin Premium**: LA Basin maintains highest prices (~50% above system average)
3. **Seasonal Variation**: Summer prices (Jul-Aug) are 2-3x winter prices
4. **Increasing Uncertainty**: Confidence intervals widen over time (more uncertainty in 2030 vs 2026)

## Dependencies

Required Python packages:
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

Install with:
```bash
pip install numpy pandas matplotlib seaborn
```

## Files Structure

```
.
├── caiso_ra_forecast.py              # Main forecast script
├── example_caiso_ra_usage.py         # Usage examples
├── CAISO_RA_README.md                # This file
├── src/
│   ├── caiso_ra_forecaster.py        # Core forecasting logic
│   └── caiso_ra_visualization.py     # Visualization module
├── forecasts/                        # Output directory for forecasts
│   ├── caiso_ra_forecast_*.csv
│   ├── caiso_ra_annual_summary_*.csv
│   └── caiso_ra_metadata_*.json
└── plots/                            # Output directory for plots
    ├── caiso_ra_multi_year_trends_*.png
    ├── caiso_ra_zone_comparison_*.png
    ├── caiso_ra_seasonal_patterns_*.png
    ├── caiso_ra_price_heatmap_*.png
    └── caiso_ra_growth_trends_*.png
```

## Use Cases

### 1. Load-Serving Entity (LSE) Planning
Estimate multi-year RA procurement costs for capacity planning and budgeting.

### 2. Generator Investment Analysis
Assess potential RA revenues for new generation or storage projects.

### 3. Market Analysis
Track RA price trends across different zones and seasons.

### 4. Regulatory Planning
Support CPUC and CAISO policy analysis with forward price projections.

### 5. Financial Modeling
Incorporate RA price forecasts into financial models for energy projects.

## Limitations and Assumptions

### Assumptions
- Base prices derived from recent market trends
- Linear compound growth model
- No structural market changes (e.g., major policy reforms)
- Normal distribution for uncertainty quantification

### Limitations
- Does not incorporate:
  - Specific planned retirements or additions
  - Extreme weather events
  - Policy changes or regulatory reforms
  - Transmission upgrades
  - Resource-specific RA values
- Simplified treatment of local vs system RA
- Historical baseline is synthetic (not actual historical data)

### Recommended Use
- Strategic planning and budgeting (2-5 year horizon)
- Scenario analysis and sensitivity studies
- Comparative analysis across zones
- Not intended for short-term trading or operational decisions

## Future Enhancements

Potential improvements:
1. Integration with CAISO OASIS API for real historical data
2. Machine learning models trained on actual RA auction results
3. Scenario-based forecasting (high/medium/low cases)
4. Incorporation of specific generator retirement schedules
5. Weather-adjusted load forecasting
6. Integration with NQC (Net Qualifying Capacity) forecasts
7. Flex RA and local RA component separation

## References

- CAISO Resource Adequacy Program: https://www.caiso.com/planning/pages/reliabilityrequirements/default.aspx
- CPUC RA Proceedings: https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/electric-power-procurement/resource-adequacy-homepage

## Support

For questions or issues:
1. Review this README and example scripts
2. Check the metadata JSON for methodology details
3. Examine the visualization plots for data validation

## License

Part of the Anthropic energy forecasting toolkit.

---

**Note**: This forecast is for planning purposes only. Actual RA prices may vary significantly based on market conditions, policy changes, and unforeseen events. Always consult with energy market experts for critical business decisions.
