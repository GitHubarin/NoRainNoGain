import streamlit as st
import leafmap.foliumap as leafmap
from PIL import Image

st.set_page_config(layout="wide", page_title="Desertification Information Hub", page_icon="🌵")

# Sidebar with author information
st.sidebar.title("About the Project")
st.sidebar.info(
    """
    **Developed by:**
    - Amarin Muelthaler
    - Luca Erdmann
    - Tim Niedermann
    
    **Special thanks to:** Qiusheng Wu for providing a foundation for our work.
    
    **GitHub Repository:** [Project Repository](https://github.com/giswqs/streamlit-geospatial)
    """
)

# Page Title
st.title("Interactive Information Hub: Addressing Desertification in Assaba, Mauritania")

# Introduction Section
st.markdown(
    """
    ### Understanding Desertification in the Assaba Region
    
    Welcome to the **Interactive Information Hub**, a data-driven platform designed to provide key stakeholders and **G20 representatives** with critical insights into **desertification trends** in the **Assaba region of Mauritania**. 
    
    This tool leverages geospatial analysis and multi-dimensional datasets to highlight environmental changes, enabling data-informed policy decisions and sustainable interventions.
    
    #### **Key Features & Insights:**
    - **Land Cover Mapping:** Identify changes in vegetation cover and land use.
    - **Precipitation:** Understand rainfall trends and anomalies in the region.
    - **Timelapse**: Visualize the progression of rainfall and population density.
    - **Gross Primary Production (GPP):** Evaluate vegetation productivity and ecosystem health.
    - **MODIS Land Cover Insights:** Track changes in land use and desert expansion.
    - **Population Density Analysis:** Assess human impact and urbanization trends.
    - **Additional Data Perspectives:** Explore other crucial environmental and climate variables.
    
    The hub aims to bridge the gap between **data science and policy-making**, equipping decision-makers with actionable intelligence for combatting desertification.
    """
)

# Display an image of the Assaba region
image_path = "assaba_region.jpg"  # Ensure this image exists in the working directory
try:
    image = Image.open(image_path)
    st.image(image, caption="The Assaba Region, Mauritania", use_container_width=True)
except:
    st.warning("Image not found. Please ensure 'assaba_region.jpg' is available in the project folder.")

# Interactive Map
st.header("Geospatial Visualization")
st.markdown("Explore the Assaba region through interactive maps and satellite imagery.")

# Set up the map centered on Assaba, Mauritania
m = leafmap.Map(center=[16.65, -11.35], zoom=7, minimap_control=True)  # Coordinates for Assaba, Mauritania
m.add_basemap("OpenTopoMap")
m.to_streamlit(height=500)

# Instructions for Users
st.header("How to Use This Tool")
st.markdown(
    """
    1. **Navigate** through different insights using the sidebar.
    2. **Explore interactive maps** by zooming and selecting different basemaps.
    3. **Analyze data layers** to uncover patterns and trends in desertification.
    4. **Use insights** for informed decision-making and policy recommendations.
    
    *This tool is part of an ongoing initiative to promote data-driven sustainability efforts.*
    """
)
