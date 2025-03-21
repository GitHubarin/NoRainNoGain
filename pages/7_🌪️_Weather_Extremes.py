import os
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

# Configure the Streamlit page layout
st.set_page_config(layout="wide", page_title="Weather Extremes", page_icon="🌪️")

# Define more readable parameter names
parameter_labels = {
    "GWETROOT": "Root Zone Soil Wetness",
    "PRECTOTCORR": "Precipitation",
    "T2M_MIN": "Minimum Temperature"
}

# Define parameter units
parameter_units = {
    "GWETROOT": "m³/m³",  # cubic meters of water per cubic meter of soil
    "PRECTOTCORR": "mm/day",  # millimeters per day
    "T2M_MIN": "°C"  # degrees Celsius
}

# Season mapping (for seasonal analysis)
seasons = {
    "Winter": ["DEC", "JAN", "FEB"],
    "Spring": ["MAR", "APR", "MAY"],
    "Summer": ["JUN", "JUL", "AUG"],
    "Fall": ["SEP", "OCT", "NOV"]
}

# Page title and description
st.title("Regional Climate Analysis")
st.markdown("""
Compare climate indicators across different regions of Assaba to understand spatial patterns and trends:

- **Root Zone Soil Wetness**: Measures soil moisture content in the root zone
- **Precipitation**: Daily precipitation amount in millimeters
- **Minimum Temperature**: Lowest daily temperature
""")

# Sidebar information and controls
st.sidebar.title("Analysis Settings")

# Define file paths for all regions
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_subdir = "data"
weather_data_subdir = "Weather_Extremes"
base_path = os.path.join(base_dir, data_subdir, weather_data_subdir)

region_files = {
    "Middle": os.path.join(base_path, "MiddleData.csv"),
    "North": os.path.join(base_path, "NorthernData.csv"),
    "South": os.path.join(base_path, "SouthernData.csv")
}

# Load all regional data
all_regions_data = {}
for region, filepath in region_files.items():
    try:
        all_regions_data[region] = pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Error loading data for {region} region: {e}")

# Only continue if we have data for all regions
if len(all_regions_data) < len(region_files):
    st.stop()

# Process the data for all regions
def process_regional_data(region_data_dict):
    """Process data from all regions into a single tidy dataframe."""
    all_data = []
    
    for region, df in region_data_dict.items():
        # Pivot the data to long format
        tidy_data = df.melt(
            id_vars=["PARAMETER", "YEAR"], 
            value_vars=["JAN", "FEB", "MAR", "APR", "MAY", "JUN", 
                        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANN"],
            var_name="Month", 
            value_name="Value"
        )
        
        # Filter to relevant parameters
        tidy_data = tidy_data[tidy_data["PARAMETER"].isin(parameter_labels.keys())]
        
        # Add region column
        tidy_data["Region"] = region
        
        # Add to our list
        all_data.append(tidy_data)
    
    # Combine all regions' data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Add a season column
    def get_season(month):
        if month == "ANN":
            return "Annual"
        for season, months in seasons.items():
            if month in months:
                return season
        return "Unknown"
    
    combined_data["Season"] = combined_data["Month"].apply(get_season)
    
    # Replace parameter codes with readable names
    combined_data["Parameter_Label"] = combined_data["PARAMETER"].map(parameter_labels)
    
    return combined_data

# Process all regions data into one tidy dataframe
tidy_regional_data = process_regional_data(all_regions_data)

# Sidebar controls for analysis type
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Regional Comparison", "Seasonal Patterns", "Trend Analysis", "Extreme Events"]
)

# Sidebar controls for parameter selection
parameter_options = list(parameter_labels.keys())
selected_parameter = st.sidebar.selectbox(
    "Select Climate Indicator", 
    parameter_options,
    format_func=lambda x: parameter_labels[x]
)

# Create different visualizations based on the selected analysis type
if analysis_type == "Regional Comparison":
    st.header(f"Regional Comparison: {parameter_labels[selected_parameter]}")
    
    # Filter data for the selected parameter
    param_data = tidy_regional_data[tidy_regional_data["PARAMETER"] == selected_parameter]
    
    # Annual average comparison across regions
    annual_data = param_data[param_data["Month"] == "ANN"]
    
    # Create the regional comparison chart
    region_comparison = alt.Chart(annual_data).mark_line(point=True).encode(
        x=alt.X("YEAR:Q", title="Year"),
        y=alt.Y("Value:Q", 
                title=f"{parameter_labels[selected_parameter]} ({parameter_units[selected_parameter]})"),
        color=alt.Color("Region:N", scale=alt.Scale(scheme="category10")),
        tooltip=["Region", "YEAR", "Value"]
    ).properties(
        height=400,
        title=f"Annual {parameter_labels[selected_parameter]} by Region"
    ).interactive()
    
    st.altair_chart(region_comparison, use_container_width=True)
    
    # Compute and display statistics
    st.subheader("Regional Statistics")
    stats_cols = st.columns(3)
    
    for i, region in enumerate(["North", "Middle", "South"]):
        region_annual = annual_data[annual_data["Region"] == region]
        
        recent_years = region_annual[region_annual["YEAR"] >= 2015]
        avg_recent = recent_years["Value"].mean()
        
        earlier_years = region_annual[region_annual["YEAR"] < 2015]
        avg_earlier = earlier_years["Value"].mean()
        
        change_pct = ((avg_recent - avg_earlier) / avg_earlier * 100) if avg_earlier != 0 else 0
        
        stats_cols[i].metric(
            f"{region} Region",
            f"{avg_recent:.2f} {parameter_units[selected_parameter]}",
            f"{change_pct:+.1f}% since 2015",
            delta_color="normal" if selected_parameter != "T2M_MIN" else "inverse"
        )
    
    # Regional differences insight
    max_region = annual_data.groupby("Region")["Value"].mean().idxmax()
    min_region = annual_data.groupby("Region")["Value"].mean().idxmin()
    
    st.info(f"""
    **Regional Insights:** 
    
    The {max_region} region consistently shows higher {parameter_labels[selected_parameter].lower()} values compared to other regions.
    The {min_region} region typically has the lowest values. This pattern reflects the geographical differences across Assaba.
    """)

elif analysis_type == "Seasonal Patterns":
    st.header(f"Seasonal Patterns: {parameter_labels[selected_parameter]}")
    
    # Filter out annual data and keep only monthly records
    seasonal_data = tidy_regional_data[
        (tidy_regional_data["PARAMETER"] == selected_parameter) & 
        (tidy_regional_data["Month"] != "ANN")
    ]
    
    # Get years range for slider
    min_year = int(seasonal_data["YEAR"].min())
    max_year = int(seasonal_data["YEAR"].max())
    
    # Year range selection
    selected_years = st.slider(
        "Select Years Range", 
        min_year, max_year, 
        (max_year - 5, max_year)
    )
    
    # Filter data for selected years
    filtered_seasonal = seasonal_data[
        (seasonal_data["YEAR"] >= selected_years[0]) & 
        (seasonal_data["YEAR"] <= selected_years[1])
    ]
    
    # Group by region, season and calculate statistics
    season_stats = filtered_seasonal.groupby(["Region", "Season", "Month"])["Value"].mean().reset_index()
    
    # Reorder months for proper display
    month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    season_stats["Month_Order"] = season_stats["Month"].apply(lambda x: month_order.index(x) if x in month_order else -1)
    season_stats = season_stats.sort_values("Month_Order")
    
    # Create the seasonal pattern chart
    seasonal_chart = alt.Chart(season_stats).mark_line().encode(
        x=alt.X("Month:N", sort=month_order, title="Month"),
        y=alt.Y("Value:Q", title=f"{parameter_labels[selected_parameter]} ({parameter_units[selected_parameter]})"),
        color=alt.Color("Region:N", scale=alt.Scale(scheme="category10")),
        strokeDash=alt.StrokeDash("Season:N"),
        tooltip=["Region", "Season", "Month", "Value"]
    ).properties(
        height=400,
        title=f"Seasonal {parameter_labels[selected_parameter]} Patterns by Region ({selected_years[0]}-{selected_years[1]})"
    ).interactive()
    
    st.altair_chart(seasonal_chart, use_container_width=True)
    
    # Seasonal insights
    st.subheader("Seasonal Insights")
    
    # Find peak seasons for each region
    peak_seasons = {}
    for region in ["North", "Middle", "South"]:
        region_data = filtered_seasonal[filtered_seasonal["Region"] == region]
        season_avgs = region_data.groupby("Season")["Value"].mean()
        peak_season = season_avgs.idxmax() if selected_parameter != "T2M_MIN" else season_avgs.idxmin()
        peak_seasons[region] = peak_season
    
    # Display as a table
    seasons_df = pd.DataFrame({
        "Region": peak_seasons.keys(),
        f"Peak Season for {parameter_labels[selected_parameter]}": peak_seasons.values()
    })
    
    st.table(seasons_df)
    
    # Additional seasonal insight
    st.info(f"""
    **Seasonal Pattern Insights:**
    
    The seasonal patterns show how {parameter_labels[selected_parameter].lower()} varies throughout the year.
    This cycle is crucial for agricultural planning and water resource management in the Assaba region.
    """)

elif analysis_type == "Trend Analysis":
    st.header(f"Long-term Trends: {parameter_labels[selected_parameter]}")
    
    # Filter data for the selected parameter, annual values only
    trend_data = tidy_regional_data[
        (tidy_regional_data["PARAMETER"] == selected_parameter) & 
        (tidy_regional_data["Month"] == "ANN")
    ]
    
    # Get the actual year range in the data
    min_year = int(trend_data["YEAR"].min())
    max_year = int(trend_data["YEAR"].max())
    
    # Create the trend chart with trendlines and fixed domain for x-axis
    base_chart = alt.Chart(trend_data).encode(
        x=alt.X("YEAR:Q", 
                title="Year", 
                scale=alt.Scale(domain=[min_year, max_year])),  # Set explicit domain
        y=alt.Y("Value:Q", 
                title=f"{parameter_labels[selected_parameter]} ({parameter_units[selected_parameter]})"),
        color=alt.Color("Region:N", scale=alt.Scale(scheme="category10"))
    )
    
    points = base_chart.mark_circle(size=60).encode(
        tooltip=["Region", "YEAR", "Value"]
    )
    
    lines = base_chart.mark_line()
    
    # Add regression lines with proper domain
    trend_lines = base_chart.transform_regression(
        "YEAR", "Value", groupby=["Region"]
    ).mark_line(strokeDash=[5, 5]).encode(
        color=alt.Color("Region:N")
    )
    
    trend_chart = (points + lines + trend_lines).properties(
        height=500,
        title=f"Long-term {parameter_labels[selected_parameter]} Trends with Regression Lines"
    ).interactive()
    
    st.altair_chart(trend_chart, use_container_width=True)
    
    # Calculate and display trend statistics
    st.subheader("Trend Analysis")
    
    # Compute slopes for each region
    trend_results = []
    
    for region in ["North", "Middle", "South"]:
        region_data = trend_data[trend_data["Region"] == region]
        region_data = region_data.sort_values("YEAR")
        
        # Simple linear regression to get slope
        years = region_data["YEAR"].values
        values = region_data["Value"].values
        
        if len(years) > 1:
            slope, intercept = np.polyfit(years, values, 1)
            # Calculate total change over the observed period
            year_span = max(years) - min(years)
            total_change = slope * year_span
            avg_value = np.mean(values)
            percent_change = (total_change / avg_value) * 100 if avg_value != 0 else 0
            
            trend_results.append({
                "Region": region,
                "Slope": slope,
                "Annual Change": f"{slope:.3f} {parameter_units[selected_parameter]}/year",
                "Percent Change": f"{percent_change:.1f}% over {year_span} years"
            })
    
    trend_df = pd.DataFrame(trend_results)
    
    # Create metrics to show the trend directions
    trend_cols = st.columns(3)
    
    for i, row in enumerate(trend_results):
        direction = "Increasing" if row["Slope"] > 0 else "Decreasing" if row["Slope"] < 0 else "Stable"
        
        # For temperature, increasing is normally shown as red (bad)
        # For precipitation and soil moisture, increasing is normally shown as green (good)
        if selected_parameter == "T2M_MIN":
            delta_color = "inverse" if row["Slope"] > 0 else "normal"
        else:
            delta_color = "normal" if row["Slope"] > 0 else "inverse"
        
        trend_cols[i].metric(
            f"{row['Region']} Region Trend",
            direction,
            row["Annual Change"],
            delta_color=delta_color
        )
    
    # Add additional trend insights
    most_change_region = max(trend_results, key=lambda x: abs(x["Slope"]))
    
    st.info(f"""
    **Trend Insights:**
    
    The {most_change_region['Region']} region shows the most significant change in {parameter_labels[selected_parameter].lower()} over time.
    This could be attributed to {
        "changing land use patterns and vegetation coverage" if selected_parameter == "GWETROOT" else 
        "shifting precipitation patterns due to climate change" if selected_parameter == "PRECTOTCORR" else
        "regional warming trends affecting minimum temperatures" if selected_parameter == "T2M_MIN" else "various factors"
    }.
    
    The analysis covers the period from {min_year} to {max_year}, showing {max_year - min_year} years of climate data.
    """)

# Fix the severity calculation in the Extreme Events section

elif analysis_type == "Extreme Events":
    st.header(f"Extreme {parameter_labels[selected_parameter]} Events")
    
    # Filter data for the selected parameter, monthly values
    monthly_data = tidy_regional_data[
        (tidy_regional_data["PARAMETER"] == selected_parameter) & 
        (tidy_regional_data["Month"] != "ANN")
    ]
    
    # Calculate thresholds for extreme events (using percentiles)
    if selected_parameter in ["GWETROOT", "PRECTOTCORR"]:
        # For these parameters, high values are extreme (floods, high moisture)
        threshold_percentile = 95
        extreme_label = f"Extremely High {parameter_labels[selected_parameter]}"
    else:
        # For temperature, low values are extreme (cold snaps)
        threshold_percentile = 5
        extreme_label = f"Extremely Low {parameter_labels[selected_parameter]}"
    
    # Calculate thresholds for each region
    thresholds = {}
    for region in ["North", "Middle", "South"]:
        region_data = monthly_data[monthly_data["Region"] == region]
        thresholds[region] = np.percentile(region_data["Value"], threshold_percentile)
    
    # Identify extreme events
    extreme_events = []
    for region, threshold in thresholds.items():
        region_data = monthly_data[monthly_data["Region"] == region].copy()
        
        if selected_parameter in ["GWETROOT", "PRECTOTCORR"]:
            # Extreme high events
            extremes = region_data[region_data["Value"] >= threshold].copy()
        else:
            # Extreme low events
            extremes = region_data[region_data["Value"] <= threshold].copy()
        
        extremes["Threshold"] = threshold
        extreme_events.append(extremes)
    
    if extreme_events:
        extreme_df = pd.concat(extreme_events)
        
        # Count extremes by year for a timeline
        extremes_by_year = extreme_df.groupby(["Region", "YEAR"]).size().reset_index(name="Extreme_Count")
        
        # Create a chart showing frequency of extreme events over time
        extreme_timeline = alt.Chart(extremes_by_year).mark_bar().encode(
            x=alt.X("YEAR:O", title="Year"),
            y=alt.Y("Extreme_Count:Q", title="Number of Extreme Events"),
            color=alt.Color("Region:N", scale=alt.Scale(scheme="category10")),
            tooltip=["Region", "YEAR", "Extreme_Count"]
        ).properties(
            height=400,
            title=f"Frequency of {extreme_label} Events by Year"
        ).interactive()
        
        st.altair_chart(extreme_timeline, use_container_width=True)
        
        # Show detailed extreme events
        st.subheader("Details of Extreme Events")
        extreme_df["Month_Year"] = extreme_df["Month"] + " " + extreme_df["YEAR"].astype(str)
        
        # Format the extreme events table
        display_df = extreme_df[["Region", "Month_Year", "Value", "Threshold"]].copy()
        
        # Calculate deviation as a numeric value
        display_df["Deviation"] = display_df["Value"] - display_df["Threshold"]
        
        # Store formatted values with units for display
        display_df["Value_Formatted"] = display_df["Value"].round(2).astype(str) + " " + parameter_units[selected_parameter]
        display_df["Threshold_Formatted"] = display_df["Threshold"].round(2).astype(str) + " " + parameter_units[selected_parameter]
        
        # Calculate severity percentile numerically first, then format as a string
        display_df["Severity_Pct"] = (abs(display_df["Deviation"]) / display_df["Threshold"] * 100).round(1)
        display_df["Severity"] = display_df["Severity_Pct"].astype(str) + "% deviation"
        
        sorted_df = display_df.sort_values(by=["Severity_Pct"], ascending=(selected_parameter == "T2M_MIN"))

        # Then subset the columns to display
        st.dataframe(
            sorted_df[["Region", "Month_Year", "Value_Formatted", "Threshold_Formatted", "Severity"]].rename(
                columns={
                    "Value_Formatted": parameter_labels[selected_parameter],
                    "Threshold_Formatted": "Threshold"
                }
            )
        )
    
        # Add insights about extreme events
        most_extreme_region = extremes_by_year.groupby("Region")["Extreme_Count"].sum().idxmax()
        
        st.info(f"""
        **Extreme Event Insights:**
        
        The {most_extreme_region} region has experienced the highest number of extreme {parameter_labels[selected_parameter].lower()} events.
        These events can have significant impacts on {
            "water availability and agricultural productivity" if selected_parameter == "GWETROOT" else 
            "flooding risk and water resource management" if selected_parameter == "PRECTOTCORR" else
            "crop growth, energy demand, and human comfort" if selected_parameter == "T2M_MIN" else "local conditions"
        }.
        """)
    else:
        st.warning("No extreme events found with the current selections.")