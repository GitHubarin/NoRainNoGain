import os
import rasterio
import numpy as np
import tempfile
import matplotlib.cm as cm
import streamlit as st
import leafmap.foliumap as leafmap
import folium
import uuid
import shutil
import gc
import atexit
import time

# Configure the Streamlit page layout
st.set_page_config(layout="wide")

# Page title and description
st.title("Annual Gross Primary Production (GPP) Data")
st.markdown("Visualize annual GPP patterns across the Assaba region from 2010 to 2023.")

# Sidebar information and controls
st.sidebar.title("Map Settings")

# Get path to data with cross-platform compatibility
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_subdir = "data"
gpp_data_subdir = "MODIS_Gross_Primary_Production_GPP"
path = os.path.join(base_dir, data_subdir, gpp_data_subdir)

# Get list of available years (from 2010 to 2023)
@st.cache_data
def get_available_years(data_path):
    """Cache the available years to avoid repeated directory scanning"""
    years = []
    for year in range(2010, 2024):
        file_path = os.path.join(data_path, f"{year}_GP.tif")
        if os.path.exists(file_path):
            years.append(year)
    return years

available_years = get_available_years(path)

# Add a selectbox for choosing a year (removed "All Years" option)
selected_year = st.sidebar.selectbox(
    "Select Year", 
    available_years,
    index=len(available_years)-1  # Default to most recent year
)

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

# Create a temp directory for visualization files
@st.cache_resource
def create_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), f"streamlit_gpp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# Simple garbage collection function
def cleanup_resources():
    """Force garbage collection to free resources"""
    gc.collect()
    time.sleep(0.1)

# Process a single year
def process_year(year, data_path, temp_dir):
    """Process a single year's data with simplified default coloring"""
    try:
        file_path = os.path.join(data_path, f"{year}_GP.tif")
        if not os.path.exists(file_path):
            return None
        
        # Open the file and read data
        with rasterio.open(file_path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()
            
            if nodata is not None:
                band[band == nodata] = np.nan
            
            # Calculate statistics
            valid_mask = ~np.isnan(band)
            valid_values = band[valid_mask]
            
            if len(valid_values) == 0:
                return None
                
            avg_gpp = float(np.mean(valid_values))
            max_gpp = float(np.max(valid_values))
            min_gpp = float(np.min(valid_values))
            
            # Use robust min/max values (percentiles) for better visualization
            p2 = float(np.percentile(valid_values, 2))
            p98 = float(np.percentile(valid_values, 98))
            
            # Default Greens colormap
            cmap = cm.get_cmap('Greens')
            
            # Normalize band to 0-1 range based on p2/p98 percentiles
            data_range = p98 - p2
            if data_range == 0:
                data_range = 1
            
            norm_band = np.zeros_like(band)
            norm_band[valid_mask] = (band[valid_mask] - p2) / data_range
            norm_band = np.clip(norm_band, 0, 1)
            
            # Apply colormap and create RGBA
            colored = cmap(norm_band)
            colored[~valid_mask, 3] = 0  # Set alpha to 0 for NaN values
            
            # Convert to 8-bit
            rgba_img = (colored * 255).astype(np.uint8)
            
            # Update profile for RGBA output
            profile.update(dtype=rasterio.uint8, count=4, nodata=0)
            
            # Save the visualization
            vis_file = os.path.join(temp_dir, f"vis_{year}_{uuid.uuid4().hex[:6]}.tif")
            
            with rasterio.open(vis_file, 'w', **profile) as dst:
                dst.write(np.moveaxis(rgba_img, 2, 0))
            
            # Return the results
            return {
                'vis_file': vis_file,
                'avg_gpp': avg_gpp,
                'max_gpp': max_gpp,
                'min_gpp': min_gpp,
                'p2': p2,
                'p98': p98
            }
        
    except Exception as e:
        st.sidebar.error(f"Error processing year {year}: {e}")
        return None

# Safe cleanup function
def cleanup_temp_dir(temp_dir):
    """Safely clean up the temp directory"""
    if not os.path.exists(temp_dir):
        return
    
    # Force garbage collection first
    cleanup_resources()
    
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass  # Silent fail

# Custom CSS for smaller metric values
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# Set up temp directory
temp_dir = create_temp_dir()
atexit.register(lambda: cleanup_temp_dir(temp_dir))

# Create a container for the map
map_container = st.container()

try:
    # Process only the selected year
    with st.spinner(f"Processing GPP data for {selected_year}..."):
        year_data = process_year(selected_year, path, temp_dir)
    
    if not year_data:
        st.warning(f"No data available for year {selected_year}.")
        st.stop()
    
    # Create the map with selected options
    with map_container:
        # Display title and metrics
        st.subheader(f"Annual GPP - {selected_year}")
        cols = st.columns(4)
        cols[0].metric("Year", selected_year)
        cols[1].metric("Average GPP", f"{year_data['avg_gpp']:.2f} g C/m²/day")
        cols[2].metric("Maximum GPP", f"{year_data['max_gpp']:.2f} g C/m²/day")
        cols[3].metric("Minimum GPP", f"{year_data['min_gpp']:.2f} g C/m²/day")
        
        # Create the map
        m = leafmap.Map(
            center=[17, -12],
            zoom=8,
            draw_control=False,
            fullscreen_control=True,
            scale_control=True
        )
        
        # Add the selected basemap
        m.add_basemap(basemap)
        
        # Define standard colors for Greens colormap
        colors = [
            '#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', 
            '#74c476', '#41ab5d', '#238b45', '#005a32'
        ]
        
        # Add colorbar legend to the map
        m.add_colorbar(
            colors=colors,
            vmin=year_data['p2'],
            vmax=year_data['p98'],
            caption="GPP (g C/m²/day)",
            position="bottomright"
        )
        
        # Add the raster layer
        m.add_raster(
            year_data['vis_file'],
            layer_name=f"Annual GPP ({selected_year})"
        )
        
        # Display the map
        m.to_streamlit(height=600)
    
    # Add data source info
    st.sidebar.info("Data source: MODIS GPP data for Assaba region, Mauritania")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.error(traceback.format_exc())

# Force final cleanup
cleanup_resources()