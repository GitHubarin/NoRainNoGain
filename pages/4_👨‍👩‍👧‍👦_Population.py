import streamlit as st
import leafmap.foliumap as leafmap
import folium
import rasterio
import numpy as np
import tempfile
import matplotlib.cm as cm
import requests
import json
import pandas as pd
import os

# Configure the Streamlit page layout
st.set_page_config(layout="wide", page_title="Population", page_icon="👨‍👩‍👧‍👦")

# Page title and description
st.title("Population Density Heatmap")
st.markdown("Visualize population density across the Assaba region for different years.")

# Sidebar information and controls
st.sidebar.title("Map Settings")

# Allow selection of year for the population density data
year = st.sidebar.selectbox("Select Year", [2010, 2015, 2020])

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
file_path = file_paths[year]

temp_path = None
title = f"Population Density ({year})"

# Function to extract population metrics for a given year
def extract_population_metrics(year_file):
    """Extract key population metrics from a GeoTIFF file."""
    try:
        with rasterio.open(year_file) as src:
            band = src.read(1, masked=True)
            band = band.filled(np.nan)
            
            # Calculate total population
            total_pop = np.nansum(band)
            
            # Calculate average density
            valid_pixels = np.sum(~np.isnan(band))
            pixel_area_sq_km = src.res[0] * src.res[1] / 1000000  # Convert to sq km
            total_area_sq_km = valid_pixels * pixel_area_sq_km
            avg_density = total_pop / total_area_sq_km if total_area_sq_km > 0 else 0
            
            # Calculate high-density area (> 75th percentile)
            p75 = np.nanpercentile(band, 75)
            high_density_area = np.sum(band > p75) * pixel_area_sq_km
            
            # Calculate percentiles for visualization
            p2, p98 = np.nanpercentile(band, (2, 98))
            
            return {
                "year": year,
                "total_population": total_pop,
                "avg_density": avg_density,
                "high_density_area": high_density_area,
                "p2": p2,
                "p98": p98,
                "band": band,
                "profile": src.profile,
                "pixel_area": pixel_area_sq_km
            }
    except Exception as e:
        st.error(f"Error processing file for year {year}: {e}")
        return None

# Calculate metrics for all years to enable trend comparison
all_years_metrics = {}
for yr in [2010, 2015, 2020]:
    all_years_metrics[yr] = extract_population_metrics(file_paths[yr])

# Create a trends dataframe
trend_data = []
for yr, metrics in all_years_metrics.items():
    if metrics:
        trend_data.append({
            "Year": yr,
            "Total Population": metrics["total_population"],
            "Average Density": metrics["avg_density"],
            "High Density Area (km²)": metrics["high_density_area"]
        })
trends_df = pd.DataFrame(trend_data)

# Process selected year for visualization
current_metrics = all_years_metrics[year]

if current_metrics:
    # Find previous year for comparison
    available_years = sorted(all_years_metrics.keys())
    year_index = available_years.index(year)
    
    # Get comparison year (previous or earliest if current is already earliest)
    if year_index > 0:
        compare_year = available_years[year_index - 1]
    else:
        compare_year = available_years[0]  # Same as current if current is earliest
    
    # Calculate changes for metrics if comparison year is different
    if year != compare_year and all_years_metrics[compare_year]:
        prev_metrics = all_years_metrics[compare_year]
        
        pop_change = current_metrics["total_population"] - prev_metrics["total_population"]
        pop_change_pct = (pop_change / prev_metrics["total_population"]) * 100 if prev_metrics["total_population"] > 0 else 0
        
        density_change = current_metrics["avg_density"] - prev_metrics["avg_density"]
        
        area_change = current_metrics["high_density_area"] - prev_metrics["high_density_area"]
    else:
        pop_change = pop_change_pct = density_change = area_change = None

    # Prepare the visualization data
    band = current_metrics["band"]
    p2 = current_metrics["p2"]
    p98 = current_metrics["p98"]
    profile = current_metrics["profile"]
    total_pop = current_metrics["total_population"]
    avg_density = current_metrics["avg_density"]
    high_density_area = current_metrics["high_density_area"]
    
    # Normalize values for visualization
    band_norm = (band - p2) / (p98 - p2)
    band_norm = np.clip(band_norm, 0, 1)
    
    # Create a mask for NaN values
    valid_mask = ~np.isnan(band_norm)
    
    # Replace NaN with 0 for colormapping
    band_norm = np.nan_to_num(band_norm, nan=0)

    # Choose a colormap
    cmap_name = st.sidebar.selectbox(
        "Select Colormap",
        ["viridis", "plasma", "inferno", "YlOrRd", "Reds", "hot", "magma"]
    )
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(band_norm)
    
    # Set alpha channel to 0 for NaN values
    colored[~valid_mask, 3] = 0
    
    # Convert to uint8
    colored_uint8 = (colored * 255).astype(np.uint8)

    # Update the profile
    profile.update(dtype=rasterio.uint8, count=4, nodata=0)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
        temp_path = tmpfile.name
    
    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(np.moveaxis(colored_uint8, -1, 0))

    # Display key population metrics with deltas
    st.subheader("Population Metrics")
    metrics_cols = st.columns(3)
    
    # Total Population with change indicator
    if pop_change is not None:
        metrics_cols[0].metric(
            "Total Population", 
            f"{int(total_pop):,}", 
            f"{pop_change:+,.0f} ({pop_change_pct:+.1f}%)"
        )
    else:
        metrics_cols[0].metric("Total Population", f"{int(total_pop):,}")
    
    # Average Density with change indicator
    if density_change is not None:
        metrics_cols[1].metric(
            "Average Density", 
            f"{round(avg_density)} people/km²", 
            f"{density_change:+.1f} people/km²"
        )
    else:
        metrics_cols[1].metric("Average Density", f"{round(avg_density)} people/km²")
    
    # High Density Area with change indicator
    if area_change is not None:
        metrics_cols[2].metric(
            "High Density Area", 
            f"{high_density_area:.0f} km²", 
            f"{area_change:+.1f} km²"
        )
    else:
        metrics_cols[2].metric("High Density Area", f"{high_density_area:.0f} km²")

    # Display reference year info
    if year != compare_year:
        st.caption(f"Changes shown compared to {compare_year}")

    # Display trend graph
    if len(trends_df) > 1:
        st.subheader("Population Trends")
        trends_tab1, trends_tab2 = st.tabs(["Chart", "Data"])
        
        with trends_tab1:
            # Create a simple line chart for population trends
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 5))
            # Plot all three metrics
            trends_df.plot(x="Year", y="Total Population", ax=ax, marker='o', label="Total Population")
            
            # Add a second y-axis for density
            ax2 = ax.twinx()
            trends_df.plot(x="Year", y="Average Density", ax=ax2, marker='s', color='green', label="Avg. Density")
            
            # Customize the plot
            ax.set_xlabel("Year")
            ax.set_ylabel("Total Population")
            ax2.set_ylabel("Average Density (people/km²)")
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
        
        with trends_tab2:
            # Display the data table
            st.dataframe(trends_df.set_index("Year").style.format({
                "Total Population": "{:,.0f}",
                "Average Density": "{:.1f}",
                "High Density Area (km²)": "{:.1f}"
            }))

    if temp_path:
        st.subheader("Interactive Map")
        # Create the map
        m = leafmap.Map(
            center=[17, -12], 
            zoom=8,
            draw_control=False,
            fullscreen_control=True,
            scale_control=True
        )
        
        # Add basemap
        basemap = st.sidebar.selectbox(
            "Select basemap layer",
            options=["OpenStreetMap", "SATELLITE", "TERRAIN", "CartoDB.DarkMatter", "CartoDB.Positron"],
            index=0
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
        
        # Add the raster data
        m.add_raster(temp_path, layer_name=title)
        
        # Optional: Add administrative boundaries
        if st.sidebar.checkbox("Show administrative boundaries", value=False):
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
                m.add_geojson(admin_url, layer_name="Country Boundaries", 
                            style={'color': 'blue', 'weight': 1, 'fillOpacity': 0})
        
        # Add a colorbar
        try:
            if cmap_name == "viridis":
                colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#9de64e', '#fde725']
            elif cmap_name == "plasma" or cmap_name == "inferno":
                colors = ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921']
            elif cmap_name in ["YlOrRd", "Reds", "hot"]:
                colors = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026']
            elif cmap_name == "magma":
                colors = ['#000004', '#3b0f70', '#8c2981', '#dd4a68', '#fd9f6c', '#fbfcbf']
            
            m.add_colorbar(
                colors=colors,
                vmin=round(p2),
                vmax=round(p98),
                caption="Population Density (people/km²)",
                position="bottomright"
            )
        except Exception as e:
            st.warning(f"Could not add colorbar: {e}")
            m.add_html(
                html='<div style="background-color:white; padding:5px; border-radius:5px; font-weight:bold;">Population Density</div>',
                position="bottomright"
            )
        
        # Render the map
        m.to_streamlit(height=600)

    else:
        st.error("Failed to process the population data. Please check your data files.")

else:
    st.error("Failed to process the population data. Please check your data files.")

# Add a reference footer
st.markdown(
    """
    ---
    🌍 **Population Density Heatmap**
    📦 [Streamlit](https://streamlit.io) • [Leafmap](https://leafmap.org) • [Rasterio](https://rasterio.readthedocs.io)
    📈 [Data Source](https://data.humdata.org/dataset/assaba-population-density-data)
    """
)