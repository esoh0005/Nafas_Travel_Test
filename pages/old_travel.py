import streamlit as st
import datetime
from datetime import timedelta
import pandas as pd
import pydeck as pdk
import numpy as np
import math
import os
import pickle
from prophet import Prophet

# Page configuration
st.set_page_config(
    page_title="NAFAS - Travel", 
    page_icon="🍃", 
    layout="wide", 
    initial_sidebar_state="auto"
)

# Load CSS
try:
    with open("styles_landing.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    st.write("Note: Style file not loaded")

# Add custom CSS
st.markdown("""
<style>
    /* Input box and control styles */
    .stDateInput > div > div,
    .stTextInput > div > div,
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Fix for select dropdown */
    .stSelectbox > div[data-baseweb="select"] > div,
    .stSelectbox > div[data-baseweb="select"],
    .stSelectbox > div[data-baseweb="popover"],
    .stSelectbox > div[data-baseweb="menu"] {
        cursor: pointer !important;
        pointer-events: auto !important;
        z-index: 9999 !important;
    }
    
    .stSelectbox {
        z-index: 9999 !important;
        position: relative !important;
    }
    
    /* Force selectbox above map */
    div[data-testid="stAppViewContainer"] > div[data-testid="stVerticalBlock"] {
        z-index: 9999 !important;
        position: relative !important;
    }
    
    /* Make clickthrough map */
    [data-testid="stDeckGlJsonChart"] div {
        pointer-events: auto !important;
    }
    
    /* Map container style */
    .map-container {
        background-color: #2a2424 !important;
        border-radius: 15px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        text-align: center !important;
        height: 400px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Journey details box */
    .journey-details {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .journey-details h4 {
        margin-top: 0;
        color: #E7CD78;
        margin-bottom: 15px;
    }
    
    .journey-details p {
        margin: 5px 0;
        padding: 0;
    }
    
    /* AQI indicator style */
    .aqi-indicator {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* AQI level colors */
    .aqi-good {background-color: #8BC34A;}
    .aqi-moderate {background-color: #FFC107;}
    .aqi-unhealthy {background-color: #FF5722;}
    
    /* Forecast panel */
    .forecast-panel {
        border-radius: 15px !important;
        padding: 20px !important;
        margin-top: 20px !important;
    }
    
    /* Title style - remove animation */
    .page-title {
        font-weight: 200 !important;
        text-align: center !important;
        letter-spacing: 5px !important;
        margin-bottom: 30px !important;
        opacity: 1 !important;
        animation: none !important;
    }
    
    /* Override global h1 animation */
    h1 {
        opacity: 1 !important;
        animation: none !important;
    }
    
    /* Title container with brown background */
    .title-container {
        background-color: #3a3232 !important; /* 调亮的咖啡色 */
        border-radius: 15px !important;
        padding: 15px !important;
        margin-top: 30px !important; /* 向下移动 */
        margin-bottom: 30px !important;
        text-align: center !important;
        display: flex !important; /* 添加flex布局以便垂直居中 */
        align-items: center !important; /* 垂直居中 */
        justify-content: center !important; /* 水平居中 */
        min-height: 80px !important; /* 确保容器有足够高度 */
    }
    
    /* Title container h1 specific styles */
    .title-container h1 {
        margin: 0 !important; /* 移除标题默认边距 */
    }
    
    /* Button style - match Home page yellow color */
    .stButton > button {
        background-color: #E7CD78 !important;
        color: black !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: "Roboto Mono", monospace !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #EFBF04 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* No animation for any text */
    * {
        animation: none !important;
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - 确保在使用前初始化
if 'search_submitted' not in st.session_state:
    st.session_state.search_submitted = False

if 'origin' not in st.session_state:
    st.session_state.origin = "Select Location"
    
if 'destination' not in st.session_state:
    st.session_state.destination = "Select Location"

# Restore original page title
st.markdown("<h1 class='page-title'>Travel Route & Air Quality</h1>", unsafe_allow_html=True)

# City coordinates database
city_coordinates = {
    "Alor Gajah": {"lat": 2.3805, "lon": 102.208, "aqi": 68, "desc": "Small town in Melaka, low industrial pollution"},
    "Bukit Jalil": {"lat": 3.0583, "lon": 101.6911, "aqi": 75, "desc": "Sports city, high traffic area"},
    "Cheras": {"lat": 3.0851, "lon": 101.7497, "aqi": 72, "desc": "Dense residential area, moderate AQI"},
    "Ipoh": {"lat": 4.5975, "lon": 101.0901, "aqi": 66, "desc": "Capital of Perak, surrounded by limestone hills"},
    "Kemaman": {"lat": 4.2447, "lon": 103.4211, "aqi": 74, "desc": "Coastal town, oil and gas industry"},
    "Kota Bharu": {"lat": 6.1254, "lon": 102.2381, "aqi": 70, "desc": "Kelantan’s capital, near Thai border"},
    "Kuala Terengganu": {"lat": 5.3299, "lon": 103.137, "aqi": 67, "desc": "Seaside city, good air quality"},
    "Kuantan": {"lat": 3.825, "lon": 103.331, "aqi": 69, "desc": "Capital of Pahang, moderate AQI"},
    "Langkawi": {"lat": 6.352, "lon": 99.7926, "aqi": 55, "desc": "Tourist island, excellent air quality"},
    "Perai": {"lat": 5.3841, "lon": 100.3973, "aqi": 78, "desc": "Industrial hub in Pulau Pinang, high pollution"},
    "Petaling Jaya": {"lat": 3.1073, "lon": 101.6067, "aqi": 65, "desc": "Residential area, better AQI"},
    "Port Dickson": {"lat": 2.5246, "lon": 101.796, "aqi": 66, "desc": "Coastal town, good sea breeze"},
    "Putrajaya": {"lat": 2.9264, "lon": 101.6964, "aqi": 60, "desc": "Administrative center, good AQI"},
    "Seremban": {"lat": 2.7258, "lon": 101.9377, "aqi": 72, "desc": "Major city, moderate AQI"},
    "Shah Alam": {"lat": 3.0836, "lon": 101.5322, "aqi": 70, "desc": "Industrial area, moderate AQI"},
    "Sungai Petani": {"lat": 5.6497, "lon": 100.4877, "aqi": 68, "desc": "Commercial hub of Kedah, moderate AQI"},
    "Tangkak": {"lat": 2.2673, "lon": 102.5457, "aqi": 64, "desc": "Johor’s northernmost town, cleaner air"},
    "Temerloh": {"lat": 3.4496, "lon": 102.4211, "aqi": 65, "desc": "Pahang's second largest town, riverine climate"}
}

# Create two column layout
input_col, map_col = st.columns([1, 2])

# Function to create route path between two points - IMPROVED FOR BETTER VISIBILITY
def create_path_data(origin, destination):
    if origin == "Select Location" or destination == "Select Location":
        return []
        
    origin_data = city_coordinates.get(origin)
    dest_data = city_coordinates.get(destination)
    
    if not origin_data or not dest_data:
        return []
    
    # Create a more realistic driving path with waypoints
    # Simulate a realistic route by adding waypoints that follow a route-like path
    
    # Calculate the direct vector between points
    dx = dest_data["lon"] - origin_data["lon"]
    dy = dest_data["lat"] - origin_data["lat"]
    
    # Determine general direction and create waypoints accordingly
    num_waypoints = 6  # Number of intermediate points
    waypoints = []
    
    # Add origin first
    waypoints.append({
        "lat": origin_data["lat"],
        "lon": origin_data["lon"]
    })
    
    # Create intermediate waypoints with road-like patterns
    # We'll create a jagged path that simulates following actual roads
    for i in range(1, num_waypoints):
        # Calculate progress along the route (0-1)
        progress = i / num_waypoints
        
        # Base position - direct interpolation
        base_lat = origin_data["lat"] + dy * progress
        base_lon = origin_data["lon"] + dx * progress
        
        # Create road-like pattern by adding perpendicular offsets
        # Alternate between horizontal and vertical segments
        if i % 2 == 1:  # Horizontal segment
            offset_lat = 0
            offset_lon = (0.1 * math.sin(progress * math.pi)) * (1-progress)
        else:  # Vertical segment
            offset_lat = (0.05 * math.sin(progress * math.pi)) * (1-progress) 
            offset_lon = 0
        
        waypoints.append({
            "lat": base_lat + offset_lat,
            "lon": base_lon + offset_lon
        })
    
    # Add destination
    waypoints.append({
        "lat": dest_data["lat"],
        "lon": dest_data["lon"]
    })
    
    return waypoints

# Malaysia view settings - FLAT MAP (no pitch)
malaysia_view = pdk.ViewState(
    latitude=3.139,
    longitude=101.6869,
    zoom=8,
    pitch=0,  # Set to 0 for flat map
    bearing=0  # Reset bearing to north
)

# Input section
with input_col:
    st.markdown("<h3 style='text-align: left;'>Plan Your Journey</h3>", unsafe_allow_html=True)
    
    # Use containers to force layering context
    with st.container():
        # Starting point selection - fixed with unique ID and session state
        origin_options = ["Select Location"] + sorted(list(city_coordinates.keys()))
        origin = st.selectbox(
            "Starting Point (Point A)", 
            options=origin_options,
            index=origin_options.index(st.session_state.origin) if st.session_state.origin in origin_options else 0,
            key="origin_selectbox"
        )
        
        # Update session state
        st.session_state.origin = origin
    
    with st.container():
        # Destination selection - fixed with unique ID and session state
        destination_options = ["Select Location"] + sorted(list(city_coordinates.keys()))
        destination = st.selectbox(
            "Destination (Point B)", 
            options=destination_options,
            index=destination_options.index(st.session_state.destination) if st.session_state.destination in destination_options else 0,
            key="destination_selectbox"
        )
        
        # Update session state
        st.session_state.destination = destination
    
    # Date selection
    travel_date = st.date_input("Date", datetime.date.today())
    
    # Journey Details - FIXED DISPLAY WITH DIRECT HTML
    distance_text = "Select starting point and destination to see journey details."
    time_text = ""
    
    # Only calculate distance if both origin and destination are selected
    if origin != "Select Location" and destination != "Select Location" and origin != destination:
        origin_data = city_coordinates.get(origin)
        dest_data = city_coordinates.get(destination)
        
        if origin_data and dest_data:
            # Calculate rough distance (in km) using Haversine formula
            R = 6371  # Earth radius in km
            lat1, lon1 = math.radians(origin_data["lat"]), math.radians(origin_data["lon"])
            lat2, lon2 = math.radians(dest_data["lat"]), math.radians(dest_data["lon"])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            # Estimate travel time (assuming average speed of 60 km/h)
            travel_time_hours = distance / 60
            travel_time_minutes = int(travel_time_hours * 60)
            
            # Format text
            distance_text = f"Distance: {distance:.1f} km"
            time_text = f"Est. Travel Time: {int(travel_time_hours)} hr {travel_time_minutes % 60} min"
    
    # Display journey details with consistent CSS
    st.markdown(f"""
    <div class="journey-details">
        <h5>Journey Details</h5>
        <p><strong>{distance_text}</strong></p>
        {f'<p><strong>{time_text}</strong></p>' if time_text else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Submit button
    if st.button("Check Air Quality"):
        st.session_state.search_submitted = True

# Map area
with map_col:
    # Prepare map data
    map_data = []
    
    # Always include origin if selected
    if origin != "Select Location" and origin in city_coordinates:
        data = city_coordinates[origin]
        map_data.append({
            "name": origin,
            "lat": data["lat"],
            "lon": data["lon"],
            "aqi": data["aqi"],
            "desc": data["desc"],
            "type": "Origin",
            "label": "A",  # Add label A for origin
            "position": [data["lon"], data["lat"]]  # Add position array format
        })
    
    # Always include destination if selected
    if destination != "Select Location" and destination in city_coordinates and destination != origin:
        data = city_coordinates[destination]
        map_data.append({
            "name": destination,
            "lat": data["lat"],
            "lon": data["lon"],
            "aqi": data["aqi"],
            "desc": data["desc"],
            "type": "Destination",
            "label": "B",  # Add label B for destination
            "position": [data["lon"], data["lat"]]  # Add position array format
        })
    
    # Create dataframe for city markers
    locations = pd.DataFrame(map_data) if map_data else pd.DataFrame(columns=["name", "lat", "lon", "aqi", "desc", "type", "label", "position"])
    
    # Create route data if origin and destination are set and are different locations
    route_data = []
    if origin != "Select Location" and destination != "Select Location" and origin != destination:
        route_data = create_path_data(origin, destination)
    
    route_df = pd.DataFrame(route_data) if route_data else pd.DataFrame(columns=["lat", "lon", "position"])
    
    # Circle markers for cities - YELLOW COLOR FOR ORIGIN AND DESTINATION (swapped colors)
    circle_layer = pdk.Layer(
        "ScatterplotLayer",
        data=locations,
        get_position="position",
        get_radius=1500,  # Size in meters
        get_fill_color=[231, 205, 120, 220],  # Yellow color (from AB labels)
        pickable=True,
        auto_highlight=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        get_line_width=500,  # Add white border width
        radius_scale=1,
        radius_min_pixels=5,
        radius_max_pixels=100,
    )
    
    # Text layer for A and B labels - ORANGE-RED COLOR with Title font style
    text_layer = pdk.Layer(
        "TextLayer",
        data=locations,
        get_position="position",
        get_text="label",  # Use "A" and "B" labels
        get_size=20,  # Increase font size to be larger than city names
        get_color=[255, 87, 34, 255],  # Orange-Red color (from circle markers)
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
        get_pixel_offset=[0, -13],  # Text above circle markers
        billboard=True,
        size_scale=1,
        size_min_pixels=5,  # Ensure visible at any zoom level
        size_max_pixels=70,  # Limit maximum size but keep large
    )
    
    # Initialize layers list
    layers = []
    
    # Add route line between origin and destination if both are selected - IMPROVED DRIVING ROUTE
    if origin != "Select Location" and destination != "Select Location" and origin != destination:
        # Get waypoints for a more realistic driving route
        waypoints = create_path_data(origin, destination)
        
        if len(waypoints) >= 2:
            # Convert waypoints to a path format suitable for PathLayer
            path_data = []
            for i in range(len(waypoints)-1):
                # We create segments for each pair of waypoints
                path_data.append({
                    "path": [[waypoints[i]["lon"], waypoints[i]["lat"]], 
                             [waypoints[i+1]["lon"], waypoints[i+1]["lat"]]],
                    "color": [255, 152, 0]  # Orange color for route
                })
            
            # Create a realistic looking car route with PathLayer
            route_layer = pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color="color",
                get_width=300,  # Width in meters
                width_min_pixels=4,  # Minimum width for visibility
                width_max_pixels=8,  # Maximum width
                rounded=True,
                joint_rounded=True,
                cap_rounded=True,
                miter_limit=2,
                billboard=False,
                pickable=False
            )
            
            # Add route layer to the map
            layers.append(route_layer)
    
    # Add all layers in correct order (route at bottom, then markers, then text)
    layers.append(circle_layer)
    layers.append(text_layer)  # Only keep the text layer, remove the white border effect
    
    # Create tooltip with no animation
    tooltip = {
        "html": """
        <div style="padding: 12px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.5);">
            <h3 style="margin-top: 0; color: #E7CD78;">{name}</h3>
            <p style="font-size: 16px;"><strong>AQI:</strong> {aqi}</p>
            <p>{desc}</p>
        </div>
        """,
        "style": {
            "backgroundColor": "rgba(42, 36, 36, 0.9)",
            "color": "white",
            "borderRadius": "8px",
            "padding": "15px",
            "fontFamily": "Roboto Mono, monospace",
            "fontSize": "14px"
        }
    }

    # Display city count only if there are markers (no flying text)
    if not locations.empty:
        st.write(f"Displaying {len(map_data)} locations on map")
    
    # Render map with flat view
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=malaysia_view,
        layers=layers,
        tooltip=tooltip
    ))
    
    # Map instructions (no animation)
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 8px; font-size: 0.8em; margin-top: 10px;">
        <p style="margin: 0; text-align: center;">Click marker to view location details | Drag to move map | Double-click to zoom</p>
    </div>
    """, unsafe_allow_html=True)

# Forecast result panel (only displayed after submitting search)
if st.session_state.search_submitted:
    # Get destination name
    dest_name = st.session_state.get('destination', 'Malaysia')
    if dest_name == "Select Location":
        dest_name = "Malaysia"
    
    # AQI score section
    # Load the model
    @st.cache_resource
    def load_model(dest):
        try:
            # Try different possible paths
            possible_paths_aqi = [
                f"prophet_model_{dest}.pkl",
                f"./prophet_model_{dest}.pkl",
                f"../prophet_model_{dest}.pkl",
                f"All_States/prophet_model_{dest}.pkl"
            ]

            for path in possible_paths_aqi:
                if os.path.exists(path):
                    with open(path, "rb") as file:
                        return pickle.load(file)

            raise FileNotFoundError(
                f"Couldn't find model file in any of the tried paths: {possible_paths_aqi}"
            )
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
        

    # Simulated data - This will come from your ML model
    dest = dest_name.replace(" ", "").lower().strip()
    model_aqi = load_model(dest)
    if model_aqi is None:
        st.error(f"Failed to load model of {dest_name}. Please check the model file.")
        st.stop()
    
     # Ensure target_date is in datetime format
    target_date = pd.to_datetime(travel_date)
    last_date = model_aqi.history['ds'].max()
    days_needed = (target_date - last_date).days + 1
    # Display model info
    st.markdown('<div class="forecast-panel">', unsafe_allow_html=True)
    st.info(
        f"Model loaded successfully. Historical data up to: {last_date.strftime('%Y-%m-%d')}"
    )
    
        
    # Create future dataframe
    future_aqi = model_aqi.make_future_dataframe(periods=days_needed, freq='D')
    print(future_aqi.columns) 

    # # Check if model was trained with temperature
    # if 'temperature' in model_aqi.extra_regressors:
    #     # Load original dataset to calculate historical averages
    #     original_data_path = f"merged_{dest}.pkl"

    #     if os.path.exists(original_data_path):
    #         df = pd.read_csv(original_data_path)
    #         df['date'] = pd.to_datetime(df['date'])

    #         # Compute historical temperature averages for each day of the year
    #         df['month_day'] = df['date'].dt.strftime('%m-%d')
    #         temp_avg = df.groupby('month_day')['temperature'].mean().to_dict()

    #         # Assign temperature values based on past averages
    #         future_aqi['month_day'] = future_aqi['ds'].dt.strftime('%m-%d')
    #         future_aqi['temperature'] = future_aqi['month_day'].map(temp_avg)

    #         # Fill missing values with the overall average temperature
    #         overall_avg_temp = df['temperature'].mean()
    #         future_aqi['temperature'].fillna(overall_avg_temp, inplace=True)

    #         # Drop temp mapping column
    #         future_aqi.drop(columns=['month_day'], inplace=True)

    #         print("Temperature values assigned for future predictions.")
    #     else:
    #         st.error(f"Data file not found for {dest_name}, cannot retrieve temperature values.")
    #         st.stop()

    # Predict AQI
    forecast_aqi = model_aqi.predict(future_aqi)
    target_aqi = forecast_aqi[forecast_aqi['ds'] == target_date]

    if target_aqi.empty:
        st.error(f"Could not find forecast for {target_date.date()}.")
        st.stop()

    # Extract AQI predictions
    predicted_aqi = target_aqi['yhat'].values[0]
    lower_bound_aqi = target_aqi['yhat_lower'].values[0]
    upper_bound_aqi = target_aqi['yhat_upper'].values[0]

    
    aqi_value = round(predicted_aqi, 2)
    
    # Determine AQI level and risk level
    if aqi_value <= 60:
        aqi_class = "aqi-good"
        risk_level = "Low"
        risk_text = "Air quality is good, suitable for all outdoor activities."
    elif aqi_value <= 80:
        aqi_class = "aqi-moderate"
        risk_level = "Medium" 
        risk_text = "Air quality is moderate, sensitive individuals may need to take precautions."
    else:
        aqi_class = "aqi-unhealthy"
        risk_level = "High"
        risk_text = "Air quality is poor, consider reducing outdoor activities."
    

    #st.markdown('<div class="forecast-panel">', unsafe_allow_html=True)    
    # Add brown title container with adjusted margin and centered text
    st.markdown(f'<div class="title-container" style="margin-top: 70px; margin-left: auto; margin-right: auto;"><h1 class="page-title" style="color: white;">Air Quality Forecast in {dest_name}</h1></div>', unsafe_allow_html=True)
  
    # Create columns layout for AQI indicator and forecast information
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown(f"""
        <div class="aqi-indicator {aqi_class}">
            <div>
                <div style="font-size: 14px; opacity: 0.8; text-align: center;">AQI</div>
                <div style="font-size: 32px;">{aqi_value}</div>
            </div>
        </div>
        <p style="text-align: center; margin-top: 10px; font-size: 16px;">Risk Level: {risk_level}</p>
        """, unsafe_allow_html=True)
    
    # 5-day forecast timeline
    st.markdown("<h3 style='text-align: center; margin: 30px 0 20px 0;'>5-Day Forecast</h3>", unsafe_allow_html=True)
    
    # Generate dates for the next 5 days
    dates = [(travel_date + timedelta(days=i)).strftime("%m/%d") for i in range(5)]
    day_names = [(travel_date + timedelta(days=i)).strftime("%a") for i in range(5)]
    
    # Simulated AQI values - This will come from your ML model
    aqi_values = [65, 75, 80, 60, 55]
    
    # Create forecast timeline
    forecast_cols = st.columns(5)
    
    for i in range(5):
        # Determine AQI level color
        if aqi_values[i] <= 60:
            aqi_color = "#8BC34A"  # Green - Good
        elif aqi_values[i] < 80:
            aqi_color = "#FFC107"  # Yellow - Moderate
        else:
            aqi_color = "#FF5722"  # Red - Unhealthy
        
        with forecast_cols[i]:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-weight: bold;">{day_names[i]}</div>
                <div style="font-size: 12px; opacity: 0.8;">{dates[i]}</div>
                <div style="width: 50px; height: 50px; border-radius: 50%; background-color: {aqi_color}; 
                      color: white; display: flex; align-items: center; justify-content: center; 
                      margin: 10px auto; font-weight: bold;">
                    {aqi_values[i]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # # Main pollutants area
    # st.markdown("<h3 style='text-align: center; margin: 30px 0 20px 0;'>Main Pollutants</h3>", unsafe_allow_html=True)
    
    # # Simulated pollutants data - This will come from your ML model
    # pollutants = {
    #     "PM2.5": "35 μg/m³",
    #     "PM10": "48 μg/m³",
    #     "O3": "85 μg/m³",
    #     "NO2": "22 μg/m³"
    # }
    
    # # Display pollutants
    # pollutant_cols = st.columns(len(pollutants))
    # for i, (pollutant, value) in enumerate(pollutants.items()):
    #     with pollutant_cols[i]:
    #         st.markdown(f"""
    #         <div style="text-align: center; background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px;">
    #             <div style="font-weight: bold; font-size: 18px;">{pollutant}</div>
    #             <div style="font-size: 16px; margin-top: 5px;">{value}</div>
    #         </div>
    #         """, unsafe_allow_html=True)
    
    # # Travel recommendations
    # st.markdown("<h3 style='text-align: center; margin: 30px 0 20px 0;'>Travel Recommendations</h3>", unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-top: 10px; text-align: center;">
    #     <p style="margin-bottom: 15px;">{risk_text}</p>
        
    #     <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 20px;">
    #         <div style="text-align: center; padding: 10px; width: 30%;">
    #             <div style="font-weight: bold; margin-bottom: 5px;">Time of Day</div>
    #             <div>{"Early morning or evening (lower pollution)" if aqi_value > 50 else "Any time is suitable"}</div>
    #         </div>
            
    #         <div style="text-align: center; padding: 10px; width: 30%;">
    #             <div style="font-weight: bold; margin-bottom: 5px;">Route</div>
    #             <div>{"Avoid high-traffic areas" if aqi_value > 50 else "All routes are suitable"}</div>
    #         </div>
            
    #         <div style="text-align: center; padding: 10px; width: 30%;">
    #             <div style="font-weight: bold; margin-bottom: 5px;">Precautions</div>
    #             <div>{"Consider wearing a mask" if aqi_value > 100 else "Standard precautions"}</div>
    #         </div>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown('</div>', unsafe_allow_html=True) 