import os
import pandas as pd
import streamlit as st
import altair as alt

# Configure the Streamlit page layout
st.set_page_config(layout="wide", page_title="Weather Extemes", page_icon="🌪️")

# Page title and description
st.title("Historical Weather Extremes Analysis")
st.markdown("""
Visualize the progression of key weather parameters over time to demonstrate trends and extremes.
- **GWETROOT**: Root Zone Soil Wetness
- **PRECTOTCORR**: Precipitation
- **T2M_MIN**: Minimum Temperature
""")

# Sidebar information and controls
st.sidebar.title("Data Selection")
region_options = ["Middle", "North", "South"]
selected_region = st.sidebar.selectbox("Select Region", region_options, index=0)

# Define file paths based on selected region
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_subdir = "data"
weather_data_subdir = "Weather_Extremes"
base_path = os.path.join(base_dir, data_subdir, weather_data_subdir)

file_mapping = {
    "Middle": os.path.join(base_path, "MiddleData.csv"),
    "North": os.path.join(base_path, "NorthernData.csv"),
    "South": os.path.join(base_path, "SouthernData.csv")
}
file_path = file_mapping[selected_region]

# Load the data
try:
    data = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Process the data
def process_data(df):
    """Process the raw data into a tidy format for visualization."""
    # Pivot the data to long format
    tidy_data = df.melt(id_vars=["PARAMETER", "YEAR"], 
                        value_vars=["JAN", "FEB", "MAR", "APR", "MAY", "JUN", 
                                    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANN"],
                        var_name="Month", value_name="Value")
    # Filter out rows where PARAMETER is not relevant
    tidy_data = tidy_data[tidy_data["PARAMETER"].isin(["GWETROOT", "PRECTOTCORR", "T2M_MIN"])]
    return tidy_data

# Process the data into a tidy format
tidy_data = process_data(data)

# Sidebar controls for parameter selection
parameter_options = ["GWETROOT", "PRECTOTCORR", "T2M_MIN"]
selected_parameters = st.sidebar.multiselect("Select Parameters", parameter_options, default=parameter_options)

# Filter data based on selected parameters
filtered_data = tidy_data[tidy_data["PARAMETER"].isin(selected_parameters)]

# Create visualizations using Altair
st.subheader("Historical Progression of Selected Parameters")
for param in selected_parameters:
    st.markdown(f"### {param}")
    param_data = filtered_data[filtered_data["PARAMETER"] == param]
    
    # Line chart for annual progression
    annual_chart = alt.Chart(param_data[param_data["Month"] == "ANN"]).mark_line().encode(
        x=alt.X("YEAR:Q", title="Year"),
        y=alt.Y("Value:Q", title=f"{param} Value"),
        color=alt.Color("PARAMETER:N", legend=None),
        tooltip=["YEAR", "Value"]
    ).properties(
        width=800,
        height=400,
        title=f"Annual Progression of {param}"
    )
    st.altair_chart(annual_chart, use_container_width=True)
    
    # Line chart for monthly progression
    monthly_chart = alt.Chart(param_data[param_data["Month"] != "ANN"]).mark_line().encode(
        x=alt.X("YEAR:O", title="Year"),
        y=alt.Y("Value:Q", title=f"{param} Value"),
        color=alt.Color("Month:N", legend=alt.Legend(title="Month")),
        tooltip=["YEAR", "Month", "Value"]
    ).properties(
        width=800,
        height=400,
        title=f"Monthly Progression of {param}"
    )
    st.altair_chart(monthly_chart, use_container_width=True)

# Add a note about the interpretation
st.sidebar.info("""
This dashboard visualizes the progression of weather extremes over time. 
- Use the sidebar to select a region and specific parameters.
- Hover over the charts to see detailed values.
""")
