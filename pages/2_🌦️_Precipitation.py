import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import os
from datetime import datetime
import io
from branca.colormap import LinearColormap
import geopandas as gpd
import tempfile
import uuid
import shutil
import requests
import json
import rasterio
import matplotlib.cm as cm
import leafmap.foliumap as leafmap
import folium

# Configure page settings
st.set_page_config(layout="wide", page_title="Precipitation Analysis", page_icon="🌦️")

# Helper function to interpret precipitation trends
def interpret_precipitation_trend(changes):
    if all(x > 0 for x in changes):
        return "There is a **consistent increasing trend** in precipitation over the selected period."
    elif all(x < 0 for x in changes):
        return "There is a **consistent decreasing trend** in precipitation over the selected period."
    elif sum(changes) > 0:
        return "The overall trend shows **increasing precipitation** despite some year-to-year fluctuations."
    elif sum(changes) < 0:
        return "The overall trend shows **decreasing precipitation** despite some year-to-year fluctuations."
    else:
        return "Precipitation shows **no clear trend** with significant year-to-year variability."

# Helper function to describe a location based on coordinates
def get_location_description(lat, lon):
    # This is a simplified function - for a real application you might want to 
    # map coordinates to actual region names or use a more sophisticated approach
    
    lat_mid = 15.5  # Adjust these values based on your actual data
    lon_mid = -12.5
    
    if lat > lat_mid and lon > lon_mid:
        return "northeastern"
    elif lat > lat_mid and lon < lon_mid:
        return "northwestern"
    elif lat < lat_mid and lon > lon_mid:
        return "southeastern"
    else:
        return "southwestern"

# Set up the page header
st.title("🌦️ Precipitation Analysis")
st.markdown("""
This dashboard provides comprehensive analysis of precipitation patterns across the Assaba region.
Explore monthly and seasonal trends, spatial distribution, and year-over-year comparisons.
""")

# Define paths for data files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets streamlit-app directory
data_dir = "data"
precipitation_dir = "Climate_Precipitation_Data"

# Combine them using os.path.join for cross-platform compatibility
PREC_DATA_DIR = os.path.join(base_dir, data_dir, precipitation_dir)
precipitation_file = os.path.join(PREC_DATA_DIR, "Enhanced_Precipitation.csv")

# Function to load and process enhanced precipitation data
def make_json_serializable(obj):
    """Convert NumPy arrays and other non-serializable objects to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [make_json_serializable(i) for i in obj]
    return obj

@st.cache_data
def load_precipitation_data(file_path):
    try:
        # Read the CSV file with pandas
        df = pd.read_csv(file_path)
        
        # Check if we have the expected columns
        expected_columns = ['YEAR', 'LAT', 'LON', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        if all(col in df.columns for col in expected_columns):
            # Convert data to appropriate types
            numeric_cols = ['YEAR', 'LAT', 'LON', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate additional fields
            months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            df['TOTAL_ANNUAL'] = df[months].sum(axis=1)
            df['WET_SEASON'] = df[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1)
            df['DRY_SEASON'] = df[['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY']].sum(axis=1)
            
            return df
        else:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            st.error(f"Missing columns in the data file: {missing_cols}")
            return None
            
    except Exception as e:
        st.error(f"Failed to process the precipitation data: {e}")
        try:
            with open(file_path, 'r') as f:
                sample = ''.join([next(f) for _ in range(5)])
            st.code(sample, language="text")
        except:
            pass
        return None

# Load precipitation data
precipitation_data = load_precipitation_data(precipitation_file)

if precipitation_data is not None:
    # Define important constants
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_map = {m_abbr: m_name for m_abbr, m_name in zip(months, month_names)}
    
    ## Get unique years and locations
    years = sorted(precipitation_data['YEAR'].unique())
    locations = precipitation_data[['LAT', 'LON']].drop_duplicates().values
    
    # Display year range information on the main page (not the sidebar)
    if len(years) > 1:
        st.subheader(f"Data from {min(years)} to {max(years)}")
    else:
        st.write(f"Data available for year: {years[0]}")
    
    # Since we are not filtering, use the complete dataset
    filtered_data = precipitation_data
    
    # ======================================================
    # Begin analysis sections
    # ======================================================
    
    # Section 1: Monthly Patterns
    st.header("Precipitation Patterns")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Monthly Analysis", "Spatial Distribution", "Year Comparison"])
    
    with tab1:
        st.subheader("Monthly Precipitation Patterns")
        
        # Calculate monthly averages by year
        monthly_avg = filtered_data.groupby('YEAR')[months].mean().reset_index()
        monthly_long = pd.melt(monthly_avg, id_vars=['YEAR'], value_vars=months, 
                              var_name='Month', value_name='Precipitation (mm)')
        monthly_long['Month Name'] = monthly_long['Month'].map(month_map)
        
        # Sort by month order
        month_order = {month: i for i, month in enumerate(months)}
        monthly_long['Month Order'] = monthly_long['Month'].map(month_order)
        monthly_long = monthly_long.sort_values(['YEAR', 'Month Order'])
        
        # Create line chart for monthly patterns
        monthly_chart = alt.Chart(monthly_long).mark_line(point=True).encode(
            x=alt.X('Month Name:N', sort=month_names, title='Month'),
            y=alt.Y('Precipitation (mm):Q', title='Precipitation (mm)'),
            color=alt.Color('YEAR:N', title='Year'),
            tooltip=[
                alt.Tooltip('Month Name:N', title='Month'),
                alt.Tooltip('YEAR:N', title='Year'),
                alt.Tooltip('Precipitation (mm):Q', title='Precipitation (mm)', format='.1f')
            ]
        ).properties(
            width=800, 
            height=400, 
            title='Monthly Precipitation Patterns by Year'
        ).interactive()
        
        st.altair_chart(monthly_chart, use_container_width=True)
        
        # Create heatmap for monthly patterns
        monthly_heat = alt.Chart(monthly_long).mark_rect().encode(
            x=alt.X('Month Name:N', sort=month_names, title='Month'),
            y=alt.Y('YEAR:O', title='Year', sort='descending'),
            color=alt.Color('Precipitation (mm):Q', 
                           scale=alt.Scale(scheme='Blues'), 
                           title='Precipitation (mm)'),
            tooltip=[
                alt.Tooltip('Month Name:N', title='Month'),
                alt.Tooltip('YEAR:O', title='Year'),
                alt.Tooltip('Precipitation (mm):Q', title='Precipitation (mm)', format='.1f')
            ]
        ).properties(
            width=800,
            height=len(years) * 25,
            title='Monthly Precipitation Heatmap'
        )
        
        st.altair_chart(monthly_heat, use_container_width=True)
        
        # Seasonal Analysis
        st.subheader("Seasonal Precipitation Analysis")
        
        # Calculate seasonal averages by year
        seasonal_data = filtered_data.groupby('YEAR')[['WET_SEASON', 'DRY_SEASON', 'TOTAL_ANNUAL']].mean().reset_index()
        seasonal_data['WET_PERCENT'] = (seasonal_data['WET_SEASON'] / seasonal_data['TOTAL_ANNUAL']) * 100
        seasonal_data['DRY_PERCENT'] = (seasonal_data['DRY_SEASON'] / seasonal_data['TOTAL_ANNUAL']) * 100
        
        # Convert to long format for visualization
        seasonal_long = pd.melt(seasonal_data, id_vars=['YEAR'], value_vars=['WET_SEASON', 'DRY_SEASON'],
                               var_name='Season', value_name='Precipitation (mm)')
        seasonal_long['Season'] = seasonal_long['Season'].replace({
            'WET_SEASON': 'Wet Season (Jun-Sep)',
            'DRY_SEASON': 'Dry Season (Oct-May)'
        })
        
        # Create bar chart for seasonal patterns
        season_chart = alt.Chart(seasonal_long).mark_bar().encode(
            x=alt.X('YEAR:O', title='Year'),
            y=alt.Y('Precipitation (mm):Q', title='Precipitation (mm)'),
            color=alt.Color('Season:N', 
                           scale=alt.Scale(
                               domain=['Wet Season (Jun-Sep)', 'Dry Season (Oct-May)'],
                               range=['#1f77b4', '#aec7e8']
                           ),
                           title='Season'),
            tooltip=[
                alt.Tooltip('YEAR:O', title='Year'),
                alt.Tooltip('Season:N', title='Season'),
                alt.Tooltip('Precipitation (mm):Q', title='Precipitation (mm)', format='.1f')
            ]
        ).properties(
            width=800,
            height=400,
            title='Seasonal Precipitation by Year'
        )
        
        st.altair_chart(season_chart, use_container_width=True)
        
        # Create line chart for wet season percentage
        percent_data = pd.melt(seasonal_data, id_vars=['YEAR'], value_vars=['WET_PERCENT', 'DRY_PERCENT'],
                              var_name='Season', value_name='Percentage (%)')
        percent_data['Season'] = percent_data['Season'].replace({
            'WET_PERCENT': 'Wet Season (Jun-Sep)',
            'DRY_PERCENT': 'Dry Season (Oct-May)'
        })
        
        percent_chart = alt.Chart(percent_data).mark_line(point=True).encode(
            x=alt.X('YEAR:O', title='Year'),
            y=alt.Y('Percentage (%):Q', title='Percentage of Annual Precipitation (%)'),
            color=alt.Color('Season:N', 
                           scale=alt.Scale(
                               domain=['Wet Season (Jun-Sep)', 'Dry Season (Oct-May)'],
                               range=['#1f77b4', '#aec7e8']
                           ),
                           title='Season'),
            tooltip=[
                alt.Tooltip('YEAR:O', title='Year'),
                alt.Tooltip('Season:N', title='Season'),
                alt.Tooltip('Percentage (%):Q', title='Percentage (%)', format='.1f')
            ]
        ).properties(
            width=800,
            height=400,
            title='Seasonal Precipitation Distribution'
        ).interactive()
        
        st.altair_chart(percent_chart, use_container_width=True)
        
        # Calculate annual total statistics
        annual_stats = filtered_data.groupby('YEAR')['TOTAL_ANNUAL'].mean().reset_index()
        avg_annual = annual_stats['TOTAL_ANNUAL'].mean()
        max_year = annual_stats.loc[annual_stats['TOTAL_ANNUAL'].idxmax()]
        min_year = annual_stats.loc[annual_stats['TOTAL_ANNUAL'].idxmin()]
        
        # Display key statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Annual Precipitation", f"{avg_annual:.1f} mm")
        col2.metric("Wettest Year", f"{int(max_year['YEAR'])} ({max_year['TOTAL_ANNUAL']:.1f} mm)")
        col3.metric("Driest Year", f"{int(min_year['YEAR'])} ({min_year['TOTAL_ANNUAL']:.1f} mm)")
        
        # Interpretation of precipitation trends
        with tab2:
            st.subheader("Spatial Distribution of Precipitation")
            
            # Year selector for spatial analysis
            selected_year = st.selectbox(
                "Select Year for Spatial Analysis",
                options=years,
                index=len(years)-1,
                key="spatial_year_selectbox"
            )
            
            # Add a selectbox for choosing a year or all years combined
            year_options = ["All Years"] + [str(x) for x in years]
            selected_year_option = st.selectbox(
                "Select Year",
                options=year_options,
                index=0,
                key="precip_year_selector"
            )
            
            # Data selection controls
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Basemap selection
                basemap_options = ["OpenStreetMap", "SATELLITE", "TERRAIN", "CartoDB.DarkMatter", "CartoDB.Positron"]
                basemap = st.selectbox(
                    "Select basemap layer",
                    options=basemap_options,
                    index=0,
                    key="basemap_selector"
                )
                
            with col2:
                # Colormap selection
                cmap_options = ["Blues", "YlGnBu", "BuGn", "GnBu", "PuBu"]
                cmap_name = st.selectbox(
                    "Select color scheme",
                    options=cmap_options,
                    index=0,
                    key="map_colormap"
                )
            
            # Optional: admin boundaries
            show_admin = st.checkbox("Show administrative boundaries", value=False, key="show_admin_boundaries")
            
            # Define the base path for your GeoTIFF files
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets streamlit-app directory
            data_dir = "data"
            precipitation_dir = "Climate_Precipitation_Data"

            # Combine them using os.path.join for cross-platform compatibility
            raster_path = os.path.join(base_dir, data_dir, precipitation_dir)

            
            # Create a temp directory for visualization files
            @st.cache_resource
            def create_temp_dir():
                temp_dir = os.path.join(tempfile.gettempdir(), f"streamlit_precip_{uuid.uuid4().hex}")
                os.makedirs(temp_dir, exist_ok=True)
                return temp_dir
            
            # Process data and create visualization files
            @st.cache_data
            def process_visualization_data(years, data_path, cmap_name, temp_dir):
                """Process years' data and create visualization files with proper styling."""
                result = {}
                
                for yr in years:
                    file_path = os.path.join(data_path, f"{yr}R.tif")
                    if not os.path.exists(file_path):
                        continue
                        
                    try:
                        with rasterio.open(file_path) as src:
                            band = src.read(1, masked=True)
                            profile = src.profile
                            
                            # Replace nodata values with NaN
                            band = band.filled(np.nan)
                            avg_precip = np.nanmean(band)
                            max_precip = np.nanmax(band)
                            min_precip = np.nanmin(band)
                            p2, p98 = np.nanpercentile(band, (2, 98))
                            
                            # Create a styled version for visualization
                            # Normalize values
                            band_norm = (band - p2) / (p98 - p2)
                            band_norm = np.clip(band_norm, 0, 1)
                            
                            # Apply colormap
                            cmap = cm.get_cmap(cmap_name)
                            
                            # Create an RGBA array, with alpha=0 for NaN values
                            colored = cmap(band_norm)
                            colored[np.isnan(band_norm), 3] = 0  # Set alpha to 0 for NaN
                            
                            # Convert to 8-bit for saving
                            colored_uint8 = (colored * 255).astype(np.uint8)
                            
                            # Update profile for 4-band RGBA output
                            profile.update(dtype=rasterio.uint8, count=4, nodata=0)
                            
                            # Save the styled version
                            vis_file = os.path.join(temp_dir, f"vis_{yr}.tif")
                            with rasterio.open(vis_file, 'w', **profile) as dst:
                                dst.write(np.moveaxis(colored_uint8, -1, 0))
                            
                            result[yr] = {
                                'vis_file': vis_file,
                                'avg_precip': avg_precip,
                                'max_precip': max_precip,
                                'min_precip': min_precip,
                                'p2': p2,
                                'p98': p98
                            }
                    except Exception as e:
                        st.warning(f"Error processing year {yr}: {str(e)}")
                
                return result
            
            # Create a map placeholder
            map_placeholder = st.empty()
            
            # Set up temp directory and process data
            temp_dir = create_temp_dir()
            try:
                # Get list of available tif years from directory
                available_years = []
                for year in range(2010, 2024):
                    file_path = os.path.join(raster_path, f"{year}R.tif")
                    if os.path.exists(file_path):
                        available_years.append(year)
                
                # Process data
                precipitation_data = process_visualization_data(available_years, raster_path, cmap_name, temp_dir)
                
                # Function to create the map
                def create_map():
                    m = leafmap.Map(
                        center=[17, -12],
                        zoom=8,
                        draw_control=False,
                        fullscreen_control=True,
                        scale_control=True
                    )
                    
                    if basemap == "OpenStreetMap":
                        m.add_basemap("OpenStreetMap")
                    elif basemap == "SATELLITE":
                        m.add_basemap("SATELLITE")
                    elif basemap == "TERRAIN":
                        m.add_basemap("TERRAIN")
                    elif basemap == "CartoDB.DarkMatter":
                        m.add_basemap("CartoDB.DarkMatter")
                    elif basemap == "CartoDB.Positron":
                        m.add_basemap("CartoDB.Positron")
                    
                    if show_admin:
                        admin_url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
                        def admin_style_function(feature):
                            return {
                                "fillColor": "#00000000",
                                "color": "#3388ff",
                                "weight": 1.5,
                                "opacity": 0.7,
                                "fillOpacity": 0,
                                "dashArray": "5, 5"
                            }
                        tooltip = folium.GeoJsonTooltip(
                            fields=["name"],
                            aliases=["Country"],
                            localize=True,
                            sticky=False,
                            labels=True,
                            style="""
                                background-color: white;
                                border: 1px solid grey;
                                border-radius: 3px;
                                box-shadow: 2px 2px 2px #00000033;
                                font-size: 12px;
                                padding: 4px;
                            """
                        )
                        try:
                            response = requests.get(admin_url)
                            admin_data = json.loads(response.text)
                            for feature in admin_data["features"]:
                                name = feature["properties"].get("name", "Unknown")
                                feature["properties"] = {"name": name}
                            
                            folium.GeoJson(
                                data=admin_data,
                                name="Country Boundaries",
                                style_function=admin_style_function,
                                tooltip=tooltip
                            ).add_to(m)
                        except Exception as e:
                            st.warning(f"Could not add admin boundaries: {e}")
                    
                    try:
                        if cmap_name == "Blues":
                            colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
                        elif cmap_name == "viridis":
                            colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#9de64e', '#fde725']
                        elif cmap_name == "plasma":
                            colors = ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921']
                        elif cmap_name in ["YlGnBu", "BuGn", "GnBu", "PuBu"]:
                            colors = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
                        elif cmap_name == "YlOrRd":
                            colors = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']
                        
                        first_year_data = precipitation_data[available_years[0]]
                        m.add_colorbar(
                            colors=colors,
                            vmin=round(first_year_data['p2']),
                            vmax=round(first_year_data['p98']),
                            caption=f"Precipitation (mm)",
                            position="bottomright"
                        )
                    except Exception as e:
                        st.warning(f"Could not add colorbar: {e}")
                    
                    return m
                
                # Create and display the map
                with map_placeholder.container():
                    m = create_map()
                    
                    # Determine which year to display:
                    if selected_year_option == "All Years":
                        # For "All Years", just display data for the most recent year.
                        display_year = available_years[-1]
                        year_label = "All Years"  # Label for "All Years"
                    else:
                        display_year = int(selected_year_option)
                        year_label = str(display_year)  # Label for a specific year
                    
                    if display_year in precipitation_data:
                        year_data = precipitation_data[display_year]
                        
                        # Add title with selected year
                        st.subheader(f"Annual Precipitation - {year_label}")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Year", year_label)
                        col2.metric("Average Precipitation", f"{year_data['avg_precip']:.1f} mm")
                        col3.metric("Maximum Precipitation", f"{year_data['max_precip']:.1f} mm")
                        col4.metric("Minimum Precipitation", f"{year_data['min_precip']:.1f} mm")
                        
                        # Use the visualization file path
                        m.add_raster(
                            year_data['vis_file'],
                            layer_name=f"Annual Precipitation ({display_year})"
                        )
                        m.to_streamlit(height=600)
                    else:
                        st.warning(f"No data available for year {display_year}.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
            # Attempt to clean up temp directory when the session ends
            def cleanup():
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except:
                    pass
            
            import atexit
            atexit.register(cleanup)

    with tab3:
        st.subheader("Year-to-Year Comparison")
        
        # Create two columns for year selection
        if len(years) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                year1 = st.selectbox("Select First Year", years, index=0, key="compare_year1")
            with col2:
                year2 = st.selectbox("Select Second Year", years, index=len(years)-1 if len(years) > 1 else 0, key="compare_year2")
            
            if year1 != year2:
                # Filter data for selected years
                data_year1 = filtered_data[filtered_data['YEAR'] == year1]
                data_year2 = filtered_data[filtered_data['YEAR'] == year2]
                
                # Calculate monthly averages for each year
                avg_year1 = data_year1[months].mean()
                avg_year2 = data_year2[months].mean()
                
                # Create comparison data frame
                comparison_data = pd.DataFrame({
                    'Month': month_names,
                    f'{year1}': avg_year1.values,
                    f'{year2}': avg_year2.values
                })
                comparison_data['Difference'] = comparison_data[f'{year2}'] - comparison_data[f'{year1}']
                comparison_data['Percent Change'] = (comparison_data['Difference'] / comparison_data[f'{year1}'] * 100).replace([np.inf, -np.inf], np.nan)
                
                # Prepare data for visualization
                comparison_long = pd.melt(comparison_data, id_vars=['Month'], value_vars=[f'{year1}', f'{year2}'],
                                        var_name='Year', value_name='Precipitation (mm)')
                
                # Add month order for sorting
                month_order = {name: i for i, name in enumerate(month_names)}
                comparison_long['Month Order'] = comparison_long['Month'].map(month_order)
                comparison_long = comparison_long.sort_values('Month Order')
                
                # Create bar chart for comparison
                comparison_chart = alt.Chart(comparison_long).mark_bar().encode(
                    x=alt.X('Month:N', sort=month_names, title='Month'),
                    y=alt.Y('Precipitation (mm):Q', title='Precipitation (mm)'),
                    color=alt.Color('Year:N', title='Year'),
                    tooltip=[
                        alt.Tooltip('Month:N', title='Month'),
                        alt.Tooltip('Year:N', title='Year'),
                        alt.Tooltip('Precipitation (mm):Q', title='Precipitation (mm)', format='.1f')
                    ]
                ).properties(
                    width=800,
                    height=400,
                    title=f'Monthly Precipitation Comparison: {year1} vs {year2}'
                )
                
                st.altair_chart(comparison_chart, use_container_width=True)
                
                # Create a difference chart
                diff_data = comparison_data[['Month', 'Difference']].copy()
                diff_data['Month Order'] = diff_data['Month'].map(month_order)
                diff_data = diff_data.sort_values('Month Order')
                
                diff_chart = alt.Chart(diff_data).mark_bar().encode(
                    x=alt.X('Month:N', sort=month_names, title='Month'),
                    y=alt.Y('Difference:Q', title=f'Difference in Precipitation (mm): {year2} - {year1}'),
                    color=alt.condition(
                        alt.datum.Difference > 0,
                        alt.value('#1a9641'),  # Green for positive
                        alt.value('#d7191c')   # Red for negative
                    ),
                    tooltip=[
                        alt.Tooltip('Month:N', title='Month'),
                        alt.Tooltip('Difference:Q', title='Difference (mm)', format='.1f')
                    ]
                ).properties(
                    width=800,
                    height=400,
                    title=f'Change in Precipitation: {year2} vs {year1}'
                )
                
                st.altair_chart(diff_chart, use_container_width=True)
                
                # Create summary table
                st.subheader(f"Summary Comparison: {year1} vs {year2}")
                
                # Calculate seasonal and annual totals
                wet_year1 = data_year1['WET_SEASON'].mean()
                wet_year2 = data_year2['WET_SEASON'].mean()
                dry_year1 = data_year1['DRY_SEASON'].mean()
                dry_year2 = data_year2['DRY_SEASON'].mean()
                total_year1 = data_year1['TOTAL_ANNUAL'].mean()
                total_year2 = data_year2['TOTAL_ANNUAL'].mean()
                
                # Create summary data frame
                summary_data = pd.DataFrame({
                    'Period': ['Wet Season (Jun-Sep)', 'Dry Season (Oct-May)', 'Annual Total'],
                    f'{year1} (mm)': [wet_year1, dry_year1, total_year1],
                    f'{year2} (mm)': [wet_year2, dry_year2, total_year2],
                    'Difference (mm)': [wet_year2 - wet_year1, dry_year2 - dry_year1, total_year2 - total_year1],
                    'Change (%)': [
                        (wet_year2 - wet_year1) / wet_year1 * 100 if wet_year1 > 0 else np.nan,
                        (dry_year2 - dry_year1) / dry_year1 * 100 if dry_year1 > 0 else np.nan,
                        (total_year2 - total_year1) / total_year1 * 100 if total_year1 > 0 else np.nan
                    ]
                })
                
                # Display summary table
                st.dataframe(summary_data.style.format({
                    f'{year1} (mm)': '{:.1f}',
                    f'{year2} (mm)': '{:.1f}',
                    'Difference (mm)': '{:.1f}',
                    'Change (%)': '{:.1f}%'
                }))
                
                # Add insights
                total_change_pct = (total_year2 - total_year1) / total_year1 * 100 if total_year1 > 0 else 0
                
                if total_change_pct > 15:
                    trend = "significantly wetter"
                    trend_color = "green"
                elif total_change_pct > 5:
                    trend = "moderately wetter"
                    trend_color = "lightgreen"
                elif total_change_pct > -5:
                    trend = "similar precipitation"
                    trend_color = "blue"
                elif total_change_pct > -15:
                    trend = "moderately drier"
                    trend_color = "orange"
                else:
                    trend = "significantly drier"
                    trend_color = "red"
                
                st.markdown(f"""
                ### Key Insights
                - {year2} was **<span style='color:{trend_color}'>{trend}</span>** compared to {year1}, with a {total_change_pct:.1f}% change in annual precipitation.
                - The most significant monthly change was in **{comparison_data.loc[comparison_data['Difference'].abs().idxmax(), 'Month']}** with a difference of **{comparison_data['Difference'].abs().max():.1f} mm**.
                """, unsafe_allow_html=True)
                
                # Display seasonal changes
                wet_change_pct = (wet_year2 - wet_year1) / wet_year1 * 100 if wet_year1 > 0 else 0
                dry_change_pct = (dry_year2 - dry_year1) / dry_year1 * 100 if dry_year1 > 0 else 0
                
                st.markdown(f"""
                ### Seasonal Changes
                - Wet season (Jun-Sep) precipitation **{'increased' if wet_change_pct > 0 else 'decreased'}** by {abs(wet_change_pct):.1f}%.
                - Dry season (Oct-May) precipitation **{'increased' if dry_change_pct > 0 else 'decreased'}** by {abs(dry_change_pct):.1f}%.
                """)
                                    
            else:
                st.warning("Please select different years for comparison.")
        else:
            st.warning("Only one year of data is available. Year-to-year comparison requires at least two years of data.")

else:
    st.error("Failed to load precipitation data. Please check the file path and format.")

# Add footer with data source information
st.markdown("---")
st.markdown("""
**Data Sources:**
- Precipitation data: NASA Power DAV 
- Time period: 2010-2022
- Region: Assaba, Mauritania
""")