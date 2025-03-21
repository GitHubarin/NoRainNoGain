import os
import numpy as np
import streamlit as st
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from IPython.display import HTML
import tempfile
import base64
from PIL import Image
import io
import altair as alt

# Configure the Streamlit page layout
st.set_page_config(layout="wide", page_title="Timelapse", page_icon="⏱️")

# Page title and description
st.title("Assaba Region Timelapse")
st.markdown("Visualize how the Assaba region has changed over time.")

# Sidebar with data selection
st.sidebar.title("Data Settings")

# Data type selection
data_type = st.sidebar.radio(
    "Select data to visualize:",
    ["Precipitation", "Population Density"]
)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets streamlit-app directory
data_dir = "data"
# Combine them using os.path.join for cross-platform compatibility
Prec_path = os.path.join(base_dir, data_dir, "Climate_Precipitation_Data")
Pop_path = os.path.join(base_dir, data_dir, "Gridded_Population_Density_Data")

# Define paths based on selection
if data_type == "Precipitation":
    data_path = Prec_path
    file_pattern = "{}R.tif"
    years_range = range(2010, 2024)
    default_colormap = "Blues"
    unit = "mm"
    title_prefix = "Annual Precipitation"
    
elif data_type == "Population Density":
    data_path = Pop_path
    file_pattern = "Assaba_Pop_{}.tif"
    years_range = [2010, 2015, 2020]
    default_colormap = "viridis"
    unit = "people/km²"
    title_prefix = "Population Density"

# Find available years
available_years = []
for year in years_range:
    file_path = os.path.join(data_path, file_pattern.format(year))
    if os.path.exists(file_path):
        available_years.append(year)

if not available_years:
    st.error(f"No data files found in {data_path}")
    st.stop()

# Animation settings
# Replace the Animation settings section with fixed values

# Remove sliders and use fixed values instead
st.sidebar.subheader("Display Settings")

# Colormap selection
colormap_groups = {
    "Blues/Greens": ["Blues", "BuGn", "GnBu", "Greens", "YlGnBu", "PuBu"],
    "Heat": ["YlOrRd", "Reds", "Oranges", "OrRd", "YlOrBr"],
    "Rainbow": ["viridis", "plasma", "inferno", "magma", "cividis"],
    "Diverging": ["coolwarm", "RdBu", "BrBG", "PiYG", "PRGn", "PuOr"],
}

# Flatten the colormap groups for selection
all_colormaps = []
for group, cmaps in colormap_groups.items():
    for cmap in cmaps:
        all_colormaps.append(cmap)

selected_colormap = st.sidebar.selectbox(
    "Select Colormap", 
    all_colormaps,
    index=all_colormaps.index(default_colormap) if default_colormap in all_colormaps else 0
)

# Fixed values instead of sliders
animation_speed = 1000  # ms between frames
animation_size = 700   # Fixed animation size optimized for clarity

# Statistics overlay (always enabled for government use case)
show_stats = True  # Always show statistics for better insights

# Helper functions for generating recommendations
def get_agricultural_recommendation(pct_change):
    if pct_change > 10:
        return "Consider water-intensive crops and expanded farming areas where suitable"
    elif pct_change > 0:
        return "Maintain current crop patterns with slight adjustments for increased rainfall"
    elif pct_change > -10:
        return "Minor adjustments to crop selection and irrigation may be needed"
    else:
        return "Consider drought-resistant crops and improved water conservation techniques"

def get_water_recommendation(pct_change):
    if pct_change > 10:
        return "Focus on flood management and water storage infrastructure to capitalize on increased rainfall"
    elif pct_change > 0:
        return "Maintain current water management approaches with moderate expansion of storage capacity"
    elif pct_change > -10:
        return "Implement water conservation measures and consider rainwater harvesting systems"
    else:
        return "Prioritize water conservation, groundwater management, and consider alternative water sources"

def get_concentration_trend(data_dict, years):
    if len(years) < 2:
        return "remained consistent"
    
    # Check if we have population_stats in our data
    if ('population_stats' in data_dict[years[0]] and 
        'population_stats' in data_dict[years[-1]]):
        first_concentration = data_dict[years[0]]['population_stats']['high_density_pct']
        last_concentration = data_dict[years[-1]]['population_stats']['high_density_pct']
        
        pct_change = ((last_concentration - first_concentration) / first_concentration * 100) if first_concentration != 0 else 0
        
        if pct_change > 15:
            return "significantly increased, indicating greater urbanization"
        elif pct_change > 5:
            return "moderately increased, suggesting gradual urbanization"
        elif pct_change > -5:
            return "remained relatively stable in distribution"
        elif pct_change > -15:
            return "moderately decreased, suggesting population spread"
        else:
            return "significantly decreased, indicating population dispersion"
    else:
        # Fallback if population_stats not available
        return "changed over time"

def get_urban_recommendation(pct_change):
    if pct_change > 20:
        return "Urgent expansion of urban facilities and housing to accommodate rapid growth"
    elif pct_change > 10:
        return "Planned urban expansion and development of new residential areas"
    elif pct_change > 0:
        return "Gradual improvement of existing urban infrastructure and moderate expansion"
    else:
        return "Focus on maintaining current urban areas and improving quality of services"

def get_infrastructure_recommendation(pct_change):
    if pct_change > 20:
        return "Significant investment needed in roads, schools, healthcare, and utilities"
    elif pct_change > 10:
        return "Moderate expansion of infrastructure with focus on high-density areas"
    elif pct_change > 0:
        return "Targeted infrastructure improvements and maintenance of existing facilities"
    else:
        return "Focus on improving quality and efficiency of existing infrastructure"
    
# Function to preprocess all data frames
# Function to preprocess all data frames
def preprocess_data(years, colormap):
    """Process all data frames upfront for the animation"""
    processed_frames = {}
    stats_data = []
    
    # Track global min/max for consistent normalization
    global_min = float('inf')
    global_max = float('-inf')
    
    # First pass: determine global min/max
    for year in years:
        file_path = os.path.join(data_path, file_pattern.format(year))
        
        try:
            with rasterio.open(file_path) as src:
                band = src.read(1, masked=True)
                data = band.filled(np.nan)
                
                # Special handling for population data
                if data_type == "Population Density":
                    # Replace negative values with NaN
                    data[data < 0] = np.nan
                    
                    # For population data, calculate percentile-based min/max
                    # to avoid extreme outliers affecting the visualization
                    if np.sum(~np.isnan(data)) > 0:
                        p02 = np.nanpercentile(data, 2)
                        p98 = np.nanpercentile(data, 98)
                        
                        # Update min/max based on percentiles instead of absolute min/max
                        global_min = min(global_min, p02)
                        global_max = max(global_max, p98)
                    else:
                        # Fallback to regular min/max if percentile can't be calculated
                        year_min = np.nanmin(data)
                        year_max = np.nanmax(data)
                        global_min = min(global_min, year_min)
                        global_max = max(global_max, year_max)
                else:
                    # Regular min/max for other data types
                    year_min = np.nanmin(data)
                    year_max = np.nanmax(data)
                    global_min = min(global_min, year_min)
                    global_max = max(global_max, year_max)
        except Exception as e:
            st.warning(f"Error processing data for {year}: {str(e)}")
    
    # Ensure sensible min value based on data type
    if data_type == "Population Density":
        global_min = max(0, global_min)  # Population density can't be negative
    elif data_type == "Precipitation":
        global_min = max(0, global_min)  # Precipitation can't be negative
    
    # Second pass: process data with consistent normalization
    for year in years:
        file_path = os.path.join(data_path, file_pattern.format(year))
        
        try:
            with rasterio.open(file_path) as src:
                band = src.read(1, masked=True)
                data = band.filled(np.nan)
                
                # Data-specific cleanup
                if data_type == "Population Density":
                    data[data < 0] = np.nan
                elif data_type == "Precipitation":
                    data[data < 0] = np.nan
                
                # Store data and metadata
                processed_frames[year] = {
                    'data': data,
                    'transform': src.transform,
                    'bounds': src.bounds,
                }
                
                # Calculate statistics
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    avg_value = np.mean(valid_data)
                    max_value = np.max(valid_data)
                    min_value = np.min(valid_data)
                    
                    # For population, calculate additional metrics
                    if data_type == "Population Density":
                        # Calculate high-density areas (> 75th percentile)
                        high_density_threshold = np.percentile(valid_data, 75)
                        high_density_count = np.sum(valid_data > high_density_threshold)
                        total_valid_count = len(valid_data)
                        high_density_pct = (high_density_count / total_valid_count) * 100
                        
                        # Add to stats
                        processed_frames[year]['population_stats'] = {
                            'high_density_threshold': high_density_threshold,
                            'high_density_pct': high_density_pct
                        }
                else:
                    avg_value = 0
                    max_value = 0
                    min_value = 0
                
                # Store statistics
                processed_frames[year]['stats'] = {
                    'mean': avg_value,
                    'max': max_value,
                    'min': min_value
                }
                
                stats_data.append({
                    'Year': year,
                    'Average': avg_value,
                    'Maximum': max_value,
                    'Minimum': min_value
                })
        except Exception as e:
            st.warning(f"Error processing data for {year}: {str(e)}")
            
    # Create stats dataframe
    stats_df = pd.DataFrame(stats_data)
    
    # Add appropriate colormaps for different data types
    if data_type == "Population Density" and selected_colormap == "viridis":
        # Consider these good alternatives for population
        st.sidebar.info("Try 'plasma', 'inferno', or 'YlOrRd' for better population density visualization")
    
    return processed_frames, stats_df, (global_min, global_max)


# Create animation frames
def create_animation_frames(data_dict, years, colormap, value_range, show_statistics=True):
    """Create individual frames for the animation"""
    frames = []
    
    for year in years:
        if year not in data_dict:
            continue
            
        frame_data = data_dict[year]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create masked array for handling NaN values
        masked_data = np.ma.masked_invalid(frame_data['data'])
        
        # Special handling for population density
        if data_type == "Population Density":
            # Use logarithmic normalization for population data to highlight concentrations
            # This makes sparse and dense areas more distinguishable
            if np.nanmax(masked_data) > 100:  # Only use log for higher density ranges
                # Avoid log(0) issues by setting a small minimum value
                min_val = max(0.1, value_range[0])
                log_norm = mcolors.LogNorm(vmin=min_val, vmax=value_range[1])
                img = ax.imshow(masked_data, cmap=colormap, norm=log_norm)
                
                # Custom colorbar ticks for log scale
                import matplotlib.ticker as ticker
                cbar = plt.colorbar(img, ax=ax)
                # Format with appropriate tick marks for log scale
                cbar.set_label(f'{data_type} ({unit})')
                ticker_locator = ticker.LogLocator(base=10, numticks=10)
                cbar.locator = ticker_locator
                cbar.update_ticks()
            else:
                # Use percentile-based normalization to enhance visibility
                # This helps when the distribution is heavily skewed
                p02 = np.nanpercentile(masked_data, 2)
                p98 = np.nanpercentile(masked_data, 98)
                norm = mcolors.Normalize(vmin=p02, vmax=p98)
                img = ax.imshow(masked_data, cmap=colormap, norm=norm)
                cbar = plt.colorbar(img, ax=ax)
                cbar.set_label(f'{data_type} ({unit})')
        else:
            # Standard normalization for precipitation and other data types
            norm = mcolors.Normalize(vmin=value_range[0], vmax=value_range[1])
            img = ax.imshow(masked_data, cmap=colormap, norm=norm)
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label(f'{data_type} ({unit})')
        
        # Add title
        ax.set_title(f"{title_prefix} - {year}", fontsize=14)
        
        # Add statistics text if enabled
        if show_statistics:
            stats = frame_data['stats']
            
            # Get data for calculating median
            data = frame_data['data']
            valid_data = data[~np.isnan(data)]
            
            # Calculate median - handle both data types the same way
            if len(valid_data) > 0:
                median_value = np.median(valid_data)  # Use standard median, not nanmedian since we already filtered
                
                stats_text = (
                    f"Average: {stats['mean']:.1f} {unit}\n"
                    f"Median: {median_value:.1f} {unit}"
                )
            else:
                # If no valid data points exist, show zeros or N/A
                stats_text = (
                    f"Average: {stats['mean']:.1f} {unit}\n"
                    f"Median: N/A"
                )
                
            # Add text box with stats
            ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, 
                verticalalignment='top', 
                horizontalalignment='left',
                fontsize=10,  
                bbox=dict(
                    boxstyle='round,pad=0.4',  
                    facecolor='white', 
                    alpha=0.8,
                    edgecolor='lightgray',  
                    linewidth=1
                ))
        
        # Hide axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        plt.close(fig)
    
    return frames

# Function to create animated GIF
def create_animated_gif(frames, duration=1000):
    """Combine frames into an animated GIF"""
    if not frames:
        return None
        
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    buf.seek(0)
    
    return buf

# Process data and create animation
with st.spinner(f"Processing {data_type} data..."):
    # Preprocess all frames
    processed_data, stats_df, value_range = preprocess_data(available_years, selected_colormap)
    
    # Create animation frames
    frames = create_animation_frames(
        processed_data, 
        available_years, 
        selected_colormap, 
        value_range, 
        show_stats
    )
    
    # Create the animation
    if frames:
        animation_buffer = create_animated_gif(frames, animation_speed)
        
        if animation_buffer:
            # Display the animation
            st.subheader("Timelapse Animation")
            
            # Convert to base64 for display
            b64 = base64.b64encode(animation_buffer.read()).decode()
            
            # Create HTML for centered, responsive animation display
            animation_html = f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/gif;base64,{b64}" width="{animation_size}px" alt="{data_type} Timelapse">
            </div>
            """
            st.markdown(animation_html, unsafe_allow_html=True)
            
            # Download button for the animation
            st.download_button(
                label="Download Animation",
                data=animation_buffer,
                file_name=f"{data_type.lower().replace(' ', '_')}_timelapse.gif",
                mime="image/gif"
            )
        else:
            st.error("Failed to create animation.")
    else:
        st.error("No frames were generated for the animation.")

# Display data trends and insights after the animation
if stats_df is not None and len(stats_df) > 1:
    st.subheader("Data Trends Over Time")
    
    # Convert 'Year' to string for better display in Altair
    stats_df['Year'] = stats_df['Year'].astype(str)
    
    # Create a color scheme based on data type
    if data_type == "Precipitation":
        color_scheme = "blues"
        domain_values = ["Average", "Maximum", "Minimum"]
        range_values = ["#3182bd", "#08519c", "#bdd7e7"]
    else:  # Population Density
        color_scheme = "yelloworangered"
        domain_values = ["Average", "Maximum", "Minimum"]
        range_values = ["#fd8d3c", "#bd0026", "#feedde"]
    
    # Prepare data for Altair in long format
    chart_data = pd.melt(
        stats_df, 
        id_vars=['Year'], 
        value_vars=['Average', 'Maximum', 'Minimum'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create main line chart with points
    line_chart = alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Value:Q', title=f'{data_type} ({unit})'),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=domain_values, range=range_values)),
        tooltip=[
            alt.Tooltip('Year:N', title='Year'),
            alt.Tooltip('Metric:N', title='Metric'),
            alt.Tooltip('Value:Q', title=f'Value ({unit})', format='.1f')
        ]
    ).properties(
        title=f'{data_type} Trends in Assaba Region',
        width=650,
        height=400
    ).interactive()
    
    # Create a second chart for year-over-year changes
    if len(stats_df) > 1:
        # Calculate year-over-year changes
        change_data = []
        
        for i in range(1, len(stats_df)):
            prev_year = stats_df.iloc[i-1]['Year']
            curr_year = stats_df.iloc[i]['Year']
            
            for metric in ['Average', 'Maximum', 'Minimum']:
                prev_value = stats_df.iloc[i-1][metric]
                curr_value = stats_df.iloc[i][metric]
                abs_change = curr_value - prev_value
                pct_change = (abs_change / prev_value * 100) if prev_value != 0 else 0
                
                change_data.append({
                    'Period': f"{prev_year}-{curr_year}",
                    'Metric': metric,
                    'Absolute Change': abs_change,
                    'Percent Change': pct_change
                })
        
        change_df = pd.DataFrame(change_data)
        
        # Create bar chart for absolute changes
        bar_chart = alt.Chart(change_df).mark_bar().encode(
            x=alt.X('Period:N', title='Time Period'),
            y=alt.Y('Absolute Change:Q', title=f'Change ({unit})'),
            color=alt.Color('Metric:N', scale=alt.Scale(domain=domain_values, range=range_values)),
            tooltip=[
                alt.Tooltip('Period:N', title='Period'),
                alt.Tooltip('Metric:N', title='Metric'),
                alt.Tooltip('Absolute Change:Q', title=f'Change ({unit})', format='.1f'),
                alt.Tooltip('Percent Change:Q', title='% Change', format='.1f')
            ]
        ).properties(
            title=f'Year-over-Year Changes in {data_type}',
            width=650,
            height=400
        ).interactive()
        
        # Display charts side by side
        col1, col2 = st.columns(2)
        col1.altair_chart(line_chart, use_container_width=True)
        col2.altair_chart(bar_chart, use_container_width=True)
        
        # Additional visualization for specific data type
        if data_type == "Population Density" and 'population_stats' in processed_data[available_years[0]]:
            # Create data for population concentration visualization
            concentration_data = []
            
            for year in available_years:
                if 'population_stats' in processed_data[year]:
                    concentration_data.append({
                        'Year': str(year),
                        'High Density Threshold': processed_data[year]['population_stats']['high_density_threshold'],
                        'High Density Area (%)': processed_data[year]['population_stats']['high_density_pct']
                    })
            
            if concentration_data:
                conc_df = pd.DataFrame(concentration_data)
                
                # Create a visualization of population concentration
                conc_chart = alt.Chart(conc_df).mark_area().encode(
                    x=alt.X('Year:N', title='Year'),
                    y=alt.Y('High Density Area (%):Q', title='High Density Area (%)'),
                    tooltip=[
                        alt.Tooltip('Year:N', title='Year'),
                        alt.Tooltip('High Density Area (%):Q', title='High Density Area (%)', format='.1f'),
                        alt.Tooltip('High Density Threshold:Q', title=f'Density Threshold ({unit})', format='.1f')
                    ]
                ).properties(
                    title='Population Concentration Over Time',
                    width=650,
                    height=400
                ).interactive()
                
                # Display annual growth rate as a bar chart
                growth_data = []
                
                for i in range(1, len(available_years)):
                    year1 = available_years[i-1]
                    year2 = available_years[i]
                    
                    avg1 = processed_data[year1]['stats']['mean']
                    avg2 = processed_data[year2]['stats']['mean']
                    
                    years_diff = year2 - year1
                    if years_diff > 0 and avg1 > 0:
                        growth_rate = ((avg2 / avg1) ** (1/years_diff) - 1) * 100
                        growth_data.append({
                            'Period': f"{year1}-{year2}",
                            'Growth Rate (%)': growth_rate
                        })
                
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)
                    
                    growth_chart = alt.Chart(growth_df).mark_bar().encode(
                        x=alt.X('Period:N', title='Period'),
                        y=alt.Y('Growth Rate (%):Q', title='Annual Growth Rate (%)'),
                        color=alt.condition(
                            alt.datum['Growth Rate (%)'] > 0,
                            alt.value('#1a9641'),  # positive - green
                            alt.value('#d7191c')   # negative - red
                        ),
                        tooltip=[
                            alt.Tooltip('Period:N', title='Period'),
                            alt.Tooltip('Growth Rate (%):Q', title='Annual Growth Rate (%)', format='.2f')
                        ]
                    ).properties(
                        title='Annual Population Growth Rate',
                        width=650,
                        height=400
                    ).interactive()
                    
                    # Display additional charts
                    col3, col4 = st.columns(2)
                    col3.altair_chart(conc_chart, use_container_width=True)
                    col4.altair_chart(growth_chart, use_container_width=True)
        
        elif data_type == "Precipitation":
            # Create seasonal/monthly distribution of precipitation if available
            # Just a placeholder example with the annual data
            
            # Create a comparison across years as a heatmap
            heatmap_data = []
            
            for i, year in enumerate(available_years):
                heatmap_data.append({
                    'Year': str(year),
                    'Value': processed_data[year]['stats']['mean'],
                    'Rank': i + 1
                })
            
            heat_df = pd.DataFrame(heatmap_data)
           
    else:
        # If only one year of data is available, just show the line chart
        st.altair_chart(line_chart, use_container_width=True)
        
    # Add the data table in an expander as before
    with st.expander("View Detailed Data"):
        # Format the data for display
        display_df = stats_df.copy()
        for col in ['Average', 'Maximum', 'Minimum']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f} {unit}")
        
        st.dataframe(display_df)