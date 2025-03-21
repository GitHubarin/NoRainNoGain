import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import folium
import streamlit as st
from PIL import Image
import io
import tempfile
import uuid
import shutil
import leafmap.foliumap as leafmap
import json
import requests
import imageio
from matplotlib.colors import ListedColormap
from datetime import datetime
import time
from branca.element import Template, MacroElement

# Configure the Streamlit page layout
st.set_page_config(layout="wide", page_title="Land Cover Mapping", page_icon="🌲")
st.title("MODIS Land Cover Visualization")
st.markdown("Visualize MODIS Land Cover data across the Assaba region.")

# Define the path to the MODIS Land Cover data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets streamlit-app directory
data_dir = "data"
land_cover_dir = "Modis_Land_Cover_Data"

# Combine them using os.path.join for cross-platform compatibility
MODIS_DATA_DIR = os.path.join(base_dir, data_dir, land_cover_dir)

# Get list of available years
modis_files = [f for f in os.listdir(MODIS_DATA_DIR) if f.endswith(".tif")]
available_years = sorted([int(f.split("LCT")[0]) for f in modis_files])

if not available_years:
    st.error(f"No MODIS Land Cover files found in {MODIS_DATA_DIR}")
    st.stop()

# MODIS Land Cover class definitions
MODIS_CLASSES = {
    0: "Water Bodies",
    1: "Evergreen Needleleaf Forests",
    2: "Evergreen Broadleaf Forests",
    3: "Deciduous Needleleaf Forests",
    4: "Deciduous Broadleaf Forests",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas", 
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    17: "Unclassified"
}

# Create a temp directory for visualization files
@st.cache_resource
def create_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), f"streamlit_modis_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# Set the colormap to viridis (fixed)
cmap_name = "viridis"

# Process MODIS data for visualization
@st.cache_data
def process_modis_data(file_path, temp_dir, attempt=0):
    """Process MODIS data and create a styled visualization file."""
    result = {}
    
    # Generate unique filename suffix to avoid conflicts
    unique_id = uuid.uuid4().hex[:8]
    
    try:
        with rasterio.open(file_path) as src:
            # Read the data and mask
            band = src.read(1)
            mask = src.read_masks(1)
            profile = src.profile
            
            # Get the nodata value if exists
            nodata = src.nodata if src.nodata is not None else 0
            
            # Create a mask for valid data
            valid_mask = (mask > 0) & (band != nodata)
            
            # Get statistics from valid data only
            valid_data = band[valid_mask]
            if len(valid_data) > 0:
                min_val = valid_data.min()
                max_val = valid_data.max()
                mean_val = valid_data.mean()
                
                # Get unique classes present in this data
                unique_classes = np.unique(valid_data).astype(int)
                
                # Create a normalized version for visualization
                # For land cover, we'll just normalize by the range
                value_range = max_val - min_val
                if value_range > 0:
                    # Normalize to 0-1 range
                    band_norm = np.zeros_like(band, dtype=float)
                    band_norm[valid_mask] = (band[valid_mask] - min_val) / value_range
                else:
                    # If there's no range, just use a constant value
                    band_norm = np.zeros_like(band, dtype=float)
                    band_norm[valid_mask] = 0.5
                
                # Apply viridis colormap
                cmap = cm.get_cmap(cmap_name)
                colored = np.zeros((band.shape[0], band.shape[1], 4), dtype=np.float32)
                
                # Apply colormap only to valid data
                colored[valid_mask] = cmap(band_norm[valid_mask])
                
                # Set alpha to 0 for invalid data
                colored[~valid_mask, 3] = 0
                
                # Convert to 8-bit for saving
                colored_uint8 = (colored * 255).astype(np.uint8)
                
                # Update profile for 4-band RGBA output
                profile.update(dtype=rasterio.uint8, count=4, nodata=None)
                
                # Generate unique filenames to avoid collisions
                base_name = os.path.basename(file_path)
                vis_file = os.path.join(temp_dir, f"modis_vis_{base_name.replace('.tif', '')}_{unique_id}.tif")
                
                # Save the styled version
                with rasterio.open(vis_file, 'w', **profile) as dst:
                    dst.write(np.moveaxis(colored_uint8, -1, 0))
                
                # Create a dict of normalized color values for each class
                class_colors = {}
                if len(unique_classes) > 0:
                    class_range = max(unique_classes) - min(unique_classes)
                    if class_range > 0:
                        for cls in unique_classes:
                            norm_val = (cls - min(unique_classes)) / class_range
                            color = cmap(norm_val)
                            # Convert to hex color
                            hex_color = '#{:02x}{:02x}{:02x}'.format(
                                int(color[0] * 255), 
                                int(color[1] * 255), 
                                int(color[2] * 255)
                            )
                            class_colors[cls] = hex_color
                    else:
                        # If only one class, use middle of colormap
                        color = cmap(0.5)
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(color[0] * 255), 
                            int(color[1] * 255), 
                            int(color[2] * 255)
                        )
                        for cls in unique_classes:
                            class_colors[cls] = hex_color
                
                result = {
                    'vis_file': vis_file,
                    'min_val': float(min_val),
                    'max_val': float(max_val),
                    'mean_val': float(mean_val),
                    'min_vis': float(min_val),
                    'max_vis': float(max_val),
                    'band': band,
                    'valid_mask': valid_mask,
                    'shape': band.shape,  # Store the shape for reference
                    'unique_classes': unique_classes,  # Store unique classes
                    'class_colors': class_colors  # Store hex colors for classes
                }
            else:
                st.warning("No valid data found in the raster file.")
    except Exception as e:
        st.error(f"Error processing MODIS data: {str(e)}")
        if attempt < 3:  # Try up to 3 times
            time.sleep(1)  # Wait a bit before retrying
            return process_modis_data(file_path, temp_dir, attempt + 1)
    
    return result

# Create a custom legend for the map
# Change the create_map_legend function to improve visibility and positioning
def create_map_legend(present_classes, class_colors):
    """Create an HTML legend to add to the map with better explanations of land cover types"""
    legend_html = """
    <div id="land-cover-legend" style="
        position: absolute; 
        bottom: 20px; 
        left: 10px; 
        width: 280px;
        height: auto;
        max-height: 300px;
        overflow-y: auto;
        background-color: white; 
        border-radius: 5px; 
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        padding: 10px;
        font-size: 12px;
        z-index: 9999;
        ">
        <div style="text-align: center; font-weight: bold; margin-bottom: 10px;">
            MODIS Land Cover Classification
        </div>
        <table style="width: 100%;">
    """
    
    class_descriptions = {
        0: "Water - Oceans, seas, lakes, reservoirs, and rivers",
        1: "Evergreen Needleleaf Forests - Dominated by evergreen conifer trees",
        2: "Evergreen Broadleaf Forests - Dominated by evergreen broadleaf trees",
        3: "Deciduous Needleleaf Forests - Dominated by deciduous conifer trees",
        4: "Deciduous Broadleaf Forests - Dominated by deciduous broadleaf trees",
        5: "Mixed Forests - Tree communities with interspersed mixtures",
        6: "Closed Shrublands - Woody vegetation less than 2m tall",
        7: "Open Shrublands - Woody vegetation with herbaceous undergrowth",
        8: "Woody Savannas - Forest canopy 30-60%, trees >2m",
        9: "Savannas - Forest canopy 10-30%, trees >2m",
        10: "Grasslands - Dominated by herbaceous annuals",
        11: "Permanent Wetlands - Permanently inundated lands",
        12: "Croplands - At least 60% cultivated",
        13: "Urban and Built-up Lands - Human infrastructure",
        14: "Cropland/Natural Vegetation Mosaics - Mix of croplands, forests, shrubland",
        15: "Permanent Snow and Ice - Snow and ice present year-round",
        16: "Barren - Exposed soil, sand, rocks with less than 10% vegetation",
        17: "Unclassified - Not categorized"
    }
    
    for cls in sorted(present_classes):
        if cls in MODIS_CLASSES:
            color = class_colors.get(cls, "#ffffff")
            description = class_descriptions.get(cls, MODIS_CLASSES[cls])
            
            legend_html += f"""
            <tr>
                <td style="padding: 3px; vertical-align: top;">
                    <div style="
                        width: 20px; 
                        height: 20px; 
                        background-color: {color}; 
                        border: 1px solid #00000055;
                        display: inline-block;
                        vertical-align: middle;
                        margin-top: 2px;
                        "></div>
                </td>
                <td style="padding: 3px; vertical-align: top;">
                    <strong>Class {cls}:</strong> {description}
                </td>
            </tr>
            """
    
    legend_html += """
        </table>
        <div style="text-align: right; font-size: 10px; margin-top: 5px; cursor: pointer;" 
             onclick="document.getElementById('land-cover-legend').style.display='none';">
            [Close]
        </div>
    </div>
    """
    
    return legend_html

# Add a simpler, more reliable Folium legend instead of the custom HTML one
def add_folium_legend(map_obj, class_colors, present_classes):
    """Add a Folium-native legend to the map for better reliability"""
    from branca.element import Figure
    from folium.map import LayerControl
    
    # Short descriptions for limited space in folium legend
    short_descriptions = {
        0: "Water Bodies",
        1: "Evergreen Needleleaf Forests",
        2: "Evergreen Broadleaf Forests",
        3: "Deciduous Needleleaf Forests",
        4: "Deciduous Broadleaf Forests",
        5: "Mixed Forests",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Croplands",
        13: "Urban and Built-up",
        14: "Cropland/Natural Veg. Mosaics",
        15: "Snow and Ice",
        16: "Barren",
        17: "Unclassified"
    }
    
    # Create a title for the legend
    title_html = '<h4 style="text-align:center; margin-bottom:10px;">Land Cover Classes</h4>'
    
    # Create a Folium Figure for the legend
    fig = Figure(width=250, height=30 * len(present_classes) + 30)
    fig.html.add_child(folium.Element(title_html))
    
    # Add each class as a separate item
    for cls in sorted(present_classes):
        if cls in MODIS_CLASSES:
            color = class_colors.get(cls, "#ffffff")
            description = short_descriptions.get(cls, MODIS_CLASSES[cls])
            
            # Add the color box and label
            label = f"<span style='background-color:{color};opacity:0.7;'>&nbsp;&nbsp;&nbsp;&nbsp;</span> <b>{cls}:</b> {description}"
            fig.html.add_child(folium.Element(f"<div style='margin-bottom:3px;'>{label}</div>"))
    
    # Add the legend to the map
    map_obj.get_root().html.add_child(fig)
    map_obj.get_root().render()
    
    # Add the legend to a Folium LayerControl to make it manageable
    legend_name = "Land Cover Classes"
    map_obj.add_child(LayerControl())
    
    return map_obj

# Create animation with improved legend
@st.cache_data
def create_animation(years_list, data_dir, temp_dir):
    """Create and process animation for all years at once with improved legend."""
    min_vals = []
    max_vals = []
    results = {}
    all_unique_classes = set()
    frames = []
    
    with st.spinner("Creating animation... This may take a moment."):
        # First pass - gather data and determine global min/max
        for yr in years_list:
            file_path = os.path.join(data_dir, f"{yr}LCT.tif")
            if os.path.exists(file_path):
                result = process_modis_data(file_path, temp_dir)
                if result:
                    results[yr] = result
                    min_vals.append(result['min_val'])
                    max_vals.append(result['max_val'])
                    # Collect all unique classes across all years
                    all_unique_classes.update(result['unique_classes'])
        
        if not results:
            return None
        
        # Get global min/max for consistent scaling
        global_min = min(min_vals) if min_vals else 0
        global_max = max(max_vals) if max_vals else 17  # Max class ID for MODIS
        
        # Sort the unique classes
        all_unique_classes = sorted(all_unique_classes)
        
        # Use a smaller size for all frames
        fig_size = (6, 5)  # inches
        dpi = 100
        
        # Calculate normalized color values for each class
        cmap = cm.get_cmap(cmap_name)
        norm_values = {}
        if len(all_unique_classes) > 0:
            class_range = max(all_unique_classes) - min(all_unique_classes)
            if class_range > 0:
                for cls in all_unique_classes:
                    norm_values[cls] = (cls - min(all_unique_classes)) / class_range
            else:
                for cls in all_unique_classes:
                    norm_values[cls] = 0.5
        
        # Create figures and save frames
        plt.ioff()  # Turn off interactive mode
        years = sorted(results.keys())
        
        # Class descriptions for the animation legend
        class_descriptions = {
            0: "Water",
            1: "Evergreen Needleleaf Forests",
            2: "Evergreen Broadleaf Forests",
            3: "Deciduous Needleleaf Forests",
            4: "Deciduous Broadleaf Forests",
            5: "Mixed Forests",
            6: "Closed Shrublands",
            7: "Open Shrublands",
            8: "Woody Savannas",
            9: "Savannas",
            10: "Grasslands",
            11: "Permanent Wetlands",
            12: "Croplands",
            13: "Urban and Built-up",
            14: "Cropland/Natural Veg. Mosaics",
            15: "Snow and Ice",
            16: "Barren",
            17: "Unclassified"
        }
        
        for yr in years:
            # Create a new figure for each frame to avoid dimension issues
            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            
            result = results[yr]
            
            # Get the band and mask
            band = result['band']
            valid_mask = result['valid_mask']
            
            # Create a masked array
            masked_band = np.ma.array(band, mask=~valid_mask)
            
            # Plot with consistent colormap
            cmap.set_bad('none')  # Transparent for masked values
            im = ax.imshow(masked_band, cmap=cmap, vmin=global_min, vmax=global_max)
            
            # Remove axes for cleaner look
            ax.axis('off')
            
            # Add year text
            ax.text(0.5, 0.95, f"MODIS Land Cover - {yr}", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=14,
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.7))
            
            # Add legend for classes present in the data
            # Get unique classes for this year
            present_classes = result['unique_classes']
            
            # Create legend handles with better labels
            legend_handles = []
            for cls in present_classes:
                if cls in class_descriptions:
                    # Get color from colormap using normalized value
                    color = cmap(norm_values[cls])
                    # Use shorter descriptions for animation legend
                    patch = mpatches.Patch(color=color, label=f"{cls}: {class_descriptions[cls]}")
                    legend_handles.append(patch)
            
            # Add legend with a reasonable number of columns
            if legend_handles:
                ncol = 1
                if len(legend_handles) > 8:
                    ncol = 2
                
                leg = ax.legend(handles=legend_handles, loc='lower center', 
                               bbox_to_anchor=(0.5, -0.05), 
                               fontsize=8, 
                               ncol=ncol,
                               framealpha=0.7)
                leg.set_zorder(20)  # Make sure legend is on top
            
            # Make layout tight
            plt.tight_layout()
            
            # Save the frame to a PNG file (more reliable than buffer)
            frame_path = os.path.join(temp_dir, f"frame_{yr}.png")
            plt.savefig(frame_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            
            # Read the saved frame as an image
            frames.append(imageio.imread(frame_path))
        
        # Close any remaining figures
        plt.close('all')
        
        # Save animation as GIF
        animation_path = os.path.join(temp_dir, f"modis_animation_{uuid.uuid4().hex}.gif")
        
        try:
            # Use a fixed FPS value
            fps = 1.0  # 1 frame per second is good for viewing changes
            
            # Create GIF with imageio
            imageio.mimsave(animation_path, frames, fps=fps, loop=0)
            return animation_path
        except Exception as e:
            st.error(f"Error creating GIF: {str(e)}")
            return None

# Sidebar controls for map appearance
st.sidebar.title("Map Settings")

# Basemap selection
basemap = st.sidebar.selectbox(
    "Select basemap layer",
    options=["OpenStreetMap", "SATELLITE", "TERRAIN", "CartoDB.DarkMatter", "CartoDB.Positron"],
    index=0
)

# Optional: admin boundaries
show_admin = st.sidebar.checkbox("Show administrative boundaries", value=False)

# Set up temp directory
temp_dir = create_temp_dir()

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["Single Year View", "Time-lapse Animation"])

try:
    # Process based on active tab
    if st.session_state.get("active_tab") is None:
        st.session_state["active_tab"] = "Single Year View"
    
    with tab1:
        # Track when this tab is active
        if not st.session_state.get("__tab_1_clicked__", False):
            st.session_state["__tab_1_clicked__"] = True
            st.session_state["active_tab"] = "Single Year View"
        
        # Process data for the selected year
        st.subheader("Select Year to Visualize")
        min_year = min(available_years)
        max_year = max(available_years)
        selected_year = st.slider("", min_year, max_year, max_year, step=1, 
                                format="%d", key="year_slider")
        
        file_path = os.path.join(MODIS_DATA_DIR, f"{selected_year}LCT.tif")
        
        # Only process the selected year's data
        modis_data = process_modis_data(file_path, temp_dir)
        
        if not modis_data:
            st.error("Failed to process MODIS data.")
            st.stop()
        
        # Display metrics above the map
        st.subheader(f"MODIS Land Cover - {selected_year}")
            
        # Create map
        m = leafmap.Map(
            center=[17, -12],
            zoom=8,
            draw_control=False,
            fullscreen_control=True,
            scale_control=True
        )
        
        # Add basemap
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
        
        # Add admin boundaries if selected
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
                st.sidebar.warning(f"Could not add admin boundaries: {e}")
                
        # Add MODIS visualization
        m.add_raster(
            modis_data['vis_file'],
            layer_name=f"MODIS Land Cover ({selected_year})"
        )
        
        # Add custom legend showing land cover classes with their colors
        legend_html = create_map_legend(
            modis_data['unique_classes'], 
            modis_data['class_colors']
        )
        
        # Add the custom HTML legend to the map
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)
        
        # Display the map
        m.to_streamlit(height=600)
        st.subheader("Land Cover Classes in Map")
        legend_cols = st.columns(3)  # Create 3 columns for compact display

        for i, cls in enumerate(sorted(modis_data['unique_classes'])):
            if cls in MODIS_CLASSES:
                color = modis_data['class_colors'].get(cls, "#ffffff")
                col_idx = i % 3  # Distribute across 3 columns
                
                # Display color box and class description using HTML
                legend_cols[col_idx].markdown(
                    f"<div style='display:flex; align-items:center; margin-bottom:8px;'>"
                    f"<div style='width:20px; height:20px; background-color:{color}; "
                    f"border:1px solid gray; margin-right:8px;'></div>"
                    f"<div><strong>Class {cls}:</strong> {MODIS_CLASSES[cls]}</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
    
    with tab2:
        # Track when this tab is active
        if not st.session_state.get("__tab_2_clicked__", False):
            st.session_state["__tab_2_clicked__"] = True
            st.session_state["active_tab"] = "Time-lapse Animation"
        
        st.subheader("Time-lapse Animation of Land Cover Changes")
        
        # Only create animation when on the animation tab
        if st.session_state["active_tab"] == "Time-lapse Animation":
            # Create animation in one step
            animation_path = create_animation(available_years, MODIS_DATA_DIR, temp_dir)
            
            if animation_path and os.path.exists(animation_path):
                st.image(animation_path, caption="MODIS Land Cover Time-lapse (2000-2022)", width=600)
                
                # Add download button for the animation
                try:
                    with open(animation_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Animation",
                            data=file,
                            file_name=f"modis_landcover_timelapse_{min(available_years)}-{max(available_years)}.gif",
                            mime="image/gif"
                        )
                except Exception as e:
                    st.error(f"Could not create download button: {e}")
            else:
                st.error("Failed to create animation. Please try again.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.error(traceback.format_exc())

# Cleanup temp directory when session ends
def cleanup():
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        pass  # Silently fail if cleanup doesn't work

import atexit
atexit.register(cleanup)