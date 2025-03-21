import streamlit as st
import leafmap.foliumap as leafmap
import folium
import rasterio
import numpy as np
import tempfile
import matplotlib.cm as cm
import geopandas as gpd
import requests
import json
import pandas as pd
import os
from io import StringIO

# Configure the Streamlit page layout
st.set_page_config(layout="wide", page_title="Population Change", page_icon="📈")

# Page title and description
st.title("Population Change Analysis")
st.markdown("""
This tool visualizes percentage population changes over time in the Assaba region. 
Compare different time periods to see areas of growth and decline.
""")

# Sidebar info - streamlined
st.sidebar.title("Settings")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the relative path to population data directory
data_subdir = "data"
pop_data_subdir = "Gridded_Population_Density_Data"

# Combine paths using os.path.join for cross-platform compatibility
path = os.path.join(base_dir, data_subdir, pop_data_subdir)

# Create file paths dictionary using os.path.join for each file
file_paths = {
    2010: os.path.join(path, "Assaba_Pop_2010.tif"),
    2015: os.path.join(path, "Assaba_Pop_2015.tif"),
    2020: os.path.join(path, "Assaba_Pop_2020.tif")
}

# Year comparison selection
compare_years = st.sidebar.selectbox(
    "Compare years",
    ["2010 to 2015", "2015 to 2020", "2010 to 2020"]
)

# Map the selection to actual year pairs
year_mapping = {
    "2010 to 2015": (2010, 2015),
    "2015 to 2020": (2015, 2020),
    "2010 to 2020": (2010, 2020)
}
start_year, end_year = year_mapping[compare_years]

# Choose colormap
cmap_options = ["RdBu_r", "RdYlGn", "coolwarm", "seismic", "BrBG"]
cmap_name = st.sidebar.selectbox(
    "Select Colormap",
    cmap_options,
    key="change_colormap"
)

# Administrative boundaries and metadata
show_admin = st.sidebar.checkbox("Show administrative boundaries", value=True)
if show_admin:
    countries_list = ["Mauritania", "Mali", "Senegal", "Morocco", "Algeria", "All West Africa"]
    selected_countries = st.sidebar.multiselect(
        "Select countries to display",
        countries_list,
        default=["Mauritania"]
    )

# Process the data
temp_path = None
title = ""

# Load both rasters
with rasterio.open(file_paths[start_year]) as src1, rasterio.open(file_paths[end_year]) as src2:
    band1 = src1.read(1, masked=True).filled(np.nan)
    band2 = src2.read(1, masked=True).filled(np.nan)
    profile = src1.profile  # Use profile from first raster
    
    # Calculate percentage change
    mask = (band1 != 0) & ~np.isnan(band1) & ~np.isnan(band2)
    change = np.zeros_like(band1)
    change[mask] = ((band2[mask] - band1[mask]) / band1[mask]) * 100
    change[~mask] = np.nan
    title = f"Percentage Population Change ({start_year} to {end_year})"
    
    # Create a mask for valid values
    valid_mask = ~np.isnan(change)
    
    # Find reasonable thresholds for change data
    if np.any(valid_mask):
        p_neg = np.nanpercentile(change[change < 0], 5) if np.any(change < 0) else 0
        p_pos = np.nanpercentile(change[change > 0], 95) if np.any(change > 0) else 0
        abs_max = max(abs(p_neg), abs(p_pos))
        
        # Normalize to -1 to 1 range
        change_norm = np.zeros_like(change)
        change_norm[valid_mask] = np.clip(change[valid_mask] / abs_max, -1, 1)
        
        # Create a custom normalization that maps -1 to 1 to the colormap
        cmap = cm.get_cmap(cmap_name)
        
        # Transform from [-1,1] to [0,1] for the colormap
        colormap_values = np.zeros_like(change_norm)
        colormap_values[valid_mask] = (change_norm[valid_mask] + 1) / 2
        
        # Apply the colormap
        colored = cmap(colormap_values)
        
        # Set alpha channel to 0 for NaN values to make them transparent
        colored[~valid_mask, 3] = 0
        
        # Apply graduated transparency for small changes
        small_change_mask = (np.abs(change_norm) < 0.1) & valid_mask
        if np.any(small_change_mask):
            colored[small_change_mask, 3] = 0.2 + 0.8 * (np.abs(change_norm[small_change_mask]) / 0.1)
        
        # Convert to uint8, preserving all 4 channels including alpha
        colored_uint8 = (colored * 255).astype(np.uint8)
        
        # Update the profile for RGBA output
        profile.update(dtype=rasterio.uint8, count=4, nodata=0)
        
        # Save the colorized raster to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
            temp_path = tmpfile.name
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(np.moveaxis(colored_uint8, -1, 0))

# Display simple statistics
if np.any(valid_mask):
    avg_change = np.nanmean(change)
    max_increase = np.nanmax(change)
    max_decrease = np.nanmin(change)
    
    # Create a simple metrics section
    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Average Change", f"{avg_change:.1f}%")
    metrics_cols[1].metric("Maximum Growth", f"{max_increase:.1f}%")
    metrics_cols[2].metric("Maximum Decline", f"{max_decrease:.1f}%")

# Country code lookup dictionary
country_codes = {
    "Mauritania": "MRT",
    "Mali": "MLI",
    "Senegal": "SEN",
    "Morocco": "MAR",
    "Algeria": "DZA"
}

# Function to load country data with economic metrics
def load_country_data(selected_countries):
    # Create a cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "country_data.geojson")
    
    # Try to load cached data first
    if os.path.exists(cache_file):
        try:
            gdf = gpd.read_file(cache_file)
            # Check if we have all the countries we need
            countries_in_file = gdf['name'].tolist() if 'name' in gdf.columns else []
            need_refresh = False
            for country in selected_countries:
                if country == "All West Africa":
                    continue
                if country not in countries_in_file:
                    need_refresh = True
                    break
            if not need_refresh:
                return gdf
        except Exception as e:
            st.sidebar.warning(f"Cache error: {e}")
            pass  # If any error, we'll download fresh data
    
    # Download country GeoJSON data
    try:
        # First try to get the Natural Earth dataset which has more economic info
        try:
            url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
            response = requests.get(url)
            world_data = gpd.read_file(StringIO(response.text))
            
            # Ensure the needed columns exist (some datasets use different names)
            if 'NAME' in world_data.columns and 'name' not in world_data.columns:
                world_data['name'] = world_data['NAME']
            
            # Add an id column if it doesn't exist
            if 'id' not in world_data.columns and 'ISO_A3' in world_data.columns:
                world_data['id'] = world_data['ISO_A3']
            
            # Keep only useful columns
            keep_cols = ['name', 'id', 'geometry']
            extra_cols = [col for col in world_data.columns if 'gdp' in col.lower() or 'pop' in col.lower() or 'income' in col.lower()]
            keep_cols.extend(extra_cols)
            
            world_data = world_data[[col for col in keep_cols if col in world_data.columns]]
            
            # Save to cache
            world_data.to_file(cache_file, driver="GeoJSON")
            return world_data
            
        except Exception as e:
            st.sidebar.warning(f"Primary source error: {e}. Trying alternative...")
            
            # Fall back to the simpler world.geo.json dataset
            url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
            response = requests.get(url)
            world_data = gpd.read_file(StringIO(response.text))
            
            # Add a name column that matches our country names
            world_data['name'] = world_data['name'].apply(lambda x: next((k for k, v in country_codes.items() if v == x), x))
            
            # Save to cache
            world_data.to_file(cache_file, driver="GeoJSON")
            return world_data
    
    except Exception as e:
        st.sidebar.warning(f"Could not load country data: {e}")
        # Create a simple fallback with just a bounding box
        bbox = {
            "type": "Feature",
            "properties": {"name": "Study Area"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-13.0, 16.0], [-11.0, 16.0], [-11.0, 18.0], [-13.0, 18.0], [-13.0, 16.0]
                ]]
            }
        }
        return gpd.GeoDataFrame.from_features([bbox])

# Create and show the map
if temp_path:
    # Create map with minimal controls
    m = leafmap.Map(
        center=[17, -12], 
        zoom=8,
        draw_control=False,
        fullscreen_control=True,
        scale_control=True
    )
    
    # Add a basic background layer
    m.add_basemap("OpenStreetMap")
    
    # Option for satellite imagery
    if st.sidebar.checkbox("Show satellite imagery", value=False):
        m.add_basemap("SATELLITE")
    
    # Add the population change layer
    m.add_raster(temp_path, layer_name=title)
    
    # Add administrative boundaries with economic data
    if show_admin:
        country_filter = []
        if "All West Africa" in selected_countries:
            country_filter = ["Mauritania", "Mali", "Senegal", "Morocco", "Algeria"]
        else:
            country_filter = selected_countries
            
        if country_filter:
            gdf = load_country_data(country_filter)
            
            # Filter to just the countries we want
            if "All West Africa" not in selected_countries:
                gdf = gdf[gdf['name'].isin(country_filter)]
            
            # Define a style function to add tooltips with economic data
            def style_function(feature):
                return {
                    "fillColor": "#ffffff00",  # Transparent fill
                    "color": "blue",
                    "weight": 1.5,
                    "fillOpacity": 0
                }
            
            def highlight_function(feature):
                return {
                    "fillColor": "#0000ff33",  # Light blue fill with some transparency
                    "color": "blue",
                    "weight": 2.5,
                    "fillOpacity": 0.3
                }
            
            # Add as a choropleth layer with tooltips
            # Replace the problematic m.add_gdf section with this fixed version:

# Add as a choropleth layer with tooltips
            if not gdf.empty:
                # Get available fields for tooltip
                available_fields = []
                economic_fields = []
                for col in gdf.columns:
                    if col not in ['geometry', 'id']:
                        available_fields.append(col)
                        # Identify economic fields to display
                        if any(term in col.lower() for term in ['gdp', 'pop', 'income', 'capita']):
                            economic_fields.append(col)
                
                # Always include name
                if 'name' not in economic_fields and 'name' in available_fields:
                    economic_fields.insert(0, 'name')
                
                # If we found economic fields, create a tooltip
                if economic_fields:
                    # Create nice aliases for the fields
                    aliases = []
                    for field in economic_fields:
                        alias = field.replace('_', ' ').title()
                        if 'gdp' in field.lower():
                            alias = 'GDP' + alias[3:]
                        if 'pop' in field.lower():
                            alias = 'Population' + alias[3:]
                        aliases.append(alias)
                    
                    # Create tooltip for GeoJSON
                    tooltip = folium.GeoJsonTooltip(
                        fields=economic_fields,
                        aliases=aliases,
                        localize=True,
                        sticky=False,
                        labels=True,
                        style="""
                            background-color: #F0EFEF;
                            border: 2px solid black;
                            border-radius: 3px;
                            box-shadow: 3px;
                        """,
                        max_width=800,
                    )
                    
                    # Convert GeoDataFrame to GeoJSON for direct use with Folium
                    gdf_json = gdf.to_json()
                    
                    # Add GeoJSON directly to avoid tooltip conflict
                    geo_json = folium.GeoJson(
                        data=gdf_json,
                        name="Countries",
                        style_function=style_function,
                        highlight_function=highlight_function,
                        tooltip=tooltip
                    )
                    
                    # Add to map
                    geo_json.add_to(m)
                else:
                    # Just add a simple country name tooltip
                    if 'name' in gdf.columns:
                        tooltip = folium.GeoJsonTooltip(
                            fields=['name'],
                            aliases=['Country'],
                            localize=True,
                            sticky=False,
                            labels=True
                        )
                        
                        # Convert GeoDataFrame to GeoJSON for direct use with Folium
                        gdf_json = gdf.to_json()
                        
                        # Add GeoJSON directly to avoid tooltip conflict
                        geo_json = folium.GeoJson(
                            data=gdf_json,
                            name="Countries",
                            style_function=style_function,
                            highlight_function=highlight_function,
                            tooltip=tooltip
                        )
                        
                        # Add to map
                        geo_json.add_to(m)
                    else:
                        # If no name column, add without tooltip
                        m.add_gdf(
                            gdf,
                            layer_name="Countries",
                            style_function=style_function,
                            highlight_function=highlight_function,
                            zoom_to_layer=False
                        )
            else:
                st.sidebar.warning("No country data available to display")
    # Add a legend to the map
    try:
        # Use predefined diverging colors that are guaranteed to be valid
        if cmap_name in ["RdBu_r", "coolwarm", "seismic"]:
            # Blue to Red colors
            colors = ['#053061', '#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b']
        else:
            # Green to Red colors (for RdYlGn and BrBG)
            colors = ['#1b7837', '#5aae61', '#a6dba0', '#f7f7f7', '#f1b6da', '#de77ae', '#c51b7d']
        
        # Add a legend with numeric values that will properly display in the map
        m.add_colorbar(
            colors=colors,
            vmin=-100,
            vmax=100,
            caption="Population Change (%)",
            position="bottomright"
        )
    except Exception as e:
        st.warning(f"Could not add colorbar: {e}")
        # Fallback: add a simple HTML label
        html_content = '''
        <div style="background-color:white; padding:5px; border-radius:5px;">
            <div style="font-weight:bold;">Population Change (%)</div>
            <div><span style="color:red;">■</span> Decrease | <span style="color:blue;">■</span> Increase</div>
        </div>
        '''
        
        m.add_html(html=html_content, position="bottomright")
    
    # Render the map to Streamlit
    m.to_streamlit(height=600)

else:
    st.error("Failed to process the population data. Please check your data files.")

# Add a reference footer
st.sidebar.info(
    """
    Data source: WorldPop - https://www.worldpop.org/
    Geographic data: Natural Earth
    """
)