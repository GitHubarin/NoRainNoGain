import os
import rasterio
import numpy as np
import tempfile
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import leafmap.foliumap as leafmap
import folium
import uuid
import shutil
import requests
import json
import time
import gc
import atexit

# Configure the Streamlit page layout
st.set_page_config(layout="wide")

# Page title and description
st.title("Annual Gross Primary Production (GPP) Data")
st.markdown("Visualize annual GPP patterns across the Assaba region from 2010 to 2023.")

# Sidebar information and controls

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the relative path to GPP data directory
data_subdir = "data"
gpp_data_subdir = "MODIS_Gross_Primary_Production_GPP"

# Combine paths using os.path.join for cross-platform compatibility
path = os.path.join(base_dir, data_subdir, gpp_data_subdir)

# Get list of available years (from 2010 to 2023)
available_years = []
for year in range(2010, 2024):
    file_path = os.path.join(path, f"{year}_GP.tif")
    if os.path.exists(file_path):
        available_years.append(year)
        
st.sidebar.title("Map Settings")
# Add a selectbox for choosing a year or all years combined.
year_options = ["All Years"] + [str(x) for x in available_years]
selected_year_option = st.sidebar.selectbox("Select Year", year_options, index=0)

# Store basemap selection in session state to keep it consistent
if "basemap" not in st.session_state:
    st.session_state.basemap = "OpenStreetMap"
basemap = st.sidebar.selectbox(
    "Select basemap layer",
    options=["OpenStreetMap", "SATELLITE", "TERRAIN", "CartoDB.DarkMatter", "CartoDB.Positron"],
    index=0,
    key="basemap_selector"
)
st.session_state.basemap = basemap

# Track open file handles
open_files = []

# Create a temp directory for visualization files
@st.cache_resource
def create_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), f"streamlit_gpp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# Fixed close_rasterio_files function that doesn't use dataset_cache
def close_rasterio_files():
    """Close all rasterio file handles to avoid permission issues."""
    # Force garbage collection
    gc.collect()
    
    # Clear the global list of open files
    global open_files
    open_files.clear()
    
    # Give the OS a little time to release files
    time.sleep(0.2)

# Process data and create visualization files with a fixed colormap
# First, let's fix the process_visualization_data function to use a direct approach

@st.cache_data
def process_visualization_data(years, data_path, temp_dir):
    """Process years' data with a simple, reliable green color gradient."""
    result = {}
    
    # Define a simple green color gradient (bright to dark)
    green_colors = [
        '#f7fcf5',  # Lightest green (almost white)
        '#e5f5e0', 
        '#c7e9c0', 
        '#a1d99b', 
        '#74c476', 
        '#41ab5d', 
        '#238b45', 
        '#005a32'   # Darkest green
    ]
    
    # Get global min/max for consistent coloring across years
    global_min = float('inf')
    global_max = float('-inf')
    year_bands = {}
    
    # First pass to determine global statistics
    for yr in years:
        file_path = os.path.join(data_path, f"{yr}_GP.tif")
        if not os.path.exists(file_path):
            continue
        
        try:
            src = None
            try:
                src = rasterio.open(file_path)
                band = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    band[band == nodata] = np.nan
                
                # Store the band for later processing
                year_bands[yr] = {
                    'band': band,
                    'profile': src.profile.copy()
                }
                
                valid_values = band[~np.isnan(band)]
                if len(valid_values) == 0:
                    continue
                
                # Update global min/max (use actual min/max instead of percentiles)
                yr_min = float(np.nanmin(valid_values))
                yr_max = float(np.nanmax(valid_values))
                global_min = min(global_min, yr_min)
                global_max = max(global_max, yr_max)
                
            finally:
                if src is not None and not src.closed:
                    src.close()
        except Exception as e:
            st.sidebar.warning(f"Error analyzing year {yr}: {str(e)}")
    
    # Second pass to create visualizations with consistent color scale
    for yr, data in year_bands.items():
        try:
            band = data['band']
            profile = data['profile']
            
            # Calculate statistics for this year
            avg_gpp = float(np.nanmean(band))
            max_gpp = float(np.nanmax(band))
            min_gpp = float(np.nanmin(band))
            
            # Scale each pixel value from 0-1 based on global min/max
            data_range = global_max - global_min
            if data_range == 0:  # Avoid division by zero
                data_range = 1
                
            # Create RGBA output image
            height, width = band.shape
            rgba_img = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Set default alpha to 0 (transparent)
            rgba_img[:, :, 3] = 0
            
            # For each non-NaN pixel, calculate color
            for i in range(height):
                for j in range(width):
                    if not np.isnan(band[i, j]):
                        # Calculate normalized value (0-1)
                        norm_val = (band[i, j] - global_min) / data_range
                        norm_val = np.clip(norm_val, 0, 1)
                        
                        # Determine which color to use from our green gradient
                        color_idx = int(norm_val * (len(green_colors) - 1))
                        color_idx = min(color_idx, len(green_colors) - 1)  # Ensure valid index
                        
                        # Get color as hex and convert to RGB
                        hex_color = green_colors[color_idx].lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        
                        # Set pixel color and alpha
                        rgba_img[i, j, 0] = r
                        rgba_img[i, j, 1] = g
                        rgba_img[i, j, 2] = b
                        rgba_img[i, j, 3] = 255  # Fully opaque
            
            # Update profile for 4-band RGBA output
            profile.update(dtype=rasterio.uint8, count=4, nodata=0)
            
            # Save the styled version with unique name
            vis_file = os.path.join(temp_dir, f"vis_{yr}_{uuid.uuid4().hex[:6]}.tif")
            
            # Open and close the destination file explicitly
            dst = None
            try:
                dst = rasterio.open(vis_file, 'w', **profile)
                dst.write(np.moveaxis(rgba_img, 2, 0))  # Reorder from HWC to CHW format
            finally:
                if dst is not None and not dst.closed:
                    dst.close()
            
            # Store file information and calculated values
            result[yr] = {
                'vis_file': vis_file,
                'avg_gpp': avg_gpp,
                'max_gpp': max_gpp,
                'min_gpp': min_gpp,
                'min_val': global_min,  # Use global values for consistency
                'max_val': global_max
            }
            
            # Force garbage collection after each year
            gc.collect()
            
        except Exception as e:
            st.sidebar.warning(f"Error processing year {yr}: {str(e)}")
    
    return result, global_min, global_max, green_colors

# Function to safely clean up the temp directory
def cleanup_temp_dir(temp_dir):
    """Safely clean up the temp directory with retries."""
    if not os.path.exists(temp_dir):
        return
    
    try:
        # First make sure all files are closed
        close_rasterio_files()
        
        # Wait a bit to let the OS release the files
        time.sleep(0.5)
        
        # Try to delete the directory
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if not os.path.exists(temp_dir):
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait longer between retries
                    time.sleep(1.0)
                    gc.collect()  # Force garbage collection
                else:
                    st.sidebar.warning(f"Failed to clean up temp directory after {max_retries} attempts: {e}")
                    # Fall back to warning but don't crash the app
    except Exception as e:
        st.sidebar.warning(f"Failed to clean up temp directory: {e}")

# Custom CSS for smaller metric values
st.markdown("""
<style>
    /* Make metric values smaller */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    /* Make metric labels smaller */
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    /* Make metric delta values smaller */
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Create a single map that will be reused
map_placeholder = st.empty()

# Set up temp directory and process data
temp_dir = create_temp_dir()

# Register cleanup function to be called at exit
atexit.register(lambda: cleanup_temp_dir(temp_dir))

try:
    # Process data with fixed color scheme
    gpp_data_result, global_min, global_max, green_colors = process_visualization_data(available_years, path, temp_dir)

    # Function to create the initial map
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
        try:
            # Use the green color scale for the colorbar
            if gpp_data_result:  # Only add colorbar if there is valid data
                m.add_colorbar(
                    colors=green_colors,
                    vmin=global_min,
                    vmax=global_max,
                    caption=f"GPP (g C/m²/day)",
                    position="bottomright"
                )
        except Exception as e:
            st.sidebar.warning(f"Could not add colorbar: {e}")
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
        if display_year in gpp_data_result:
            year_data = gpp_data_result[display_year]
            # Add title with selected year
            st.subheader(f"Annual GPP - {year_label}")
            # Display metrics in smaller size (CSS applied above)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Year", year_label)
            col2.metric("Average GPP", f"{year_data['avg_gpp']:.2f} g C/m²/day")
            col3.metric("Maximum GPP", f"{year_data['max_gpp']:.2f} g C/m²/day")
            col4.metric("Minimum GPP", f"{year_data['min_gpp']:.2f} g C/m²/day")
            
            # Use the visualization file path
            m.add_raster(
                year_data['vis_file'],
                layer_name=f"Annual GPP ({display_year})"
            )
            m.to_streamlit(height=600)
            
            # Add a manual legend below the map for reference
        else:
            st.warning(f"No data available for year {display_year}.")
    st.sidebar.info("Data source: MODIS Gross Primary Production (GPP) data for Assaba region, Mauritania")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
finally:
    # Explicitly close all file handles first
    close_rasterio_files()