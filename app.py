# -*- coding: utf-8 -*-
import streamlit as st
import cv2 
import tempfile
import numpy as np
import time
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="ATRICS: AI-based Traffic Regulation & Intelligent Control System",
    page_icon="🚦"
)

# --- Functions with Streamlit Caching ---
# @st.cache_resource is used to cache the YOLO model so it's loaded only once.
# This prevents the app from reloading the model on every user interaction,
# which would be very slow.
@st.cache_resource
def load_yolo_model():
    """Load the pre-trained YOLOv8 model."""
    # Ensure you have 'yolov8n.pt' or your custom model file in the same directory.
    model = YOLO('yolov8n.pt')
    return model

# --- Main Dashboard Layout ---
st.title("ATRICS Dashboard")

st.markdown("""
Welcome to the ATRICS dashboard. This tool provides real-time traffic analysis using
AI-based object detection and can be integrated with traffic simulations.
""")

# Create two columns for the dashboard layout
col1, col2 = st.columns([2, 1])

# --- Column 1: Video and Simulation Display ---
with col1:
    st.header("Traffic Simulation & Analysis")
    video_file = st.file_uploader("Upload a video file for analysis", type=['mp4', 'mov', 'avi'])

    if video_file:
        # The following lines must be indented to be part of the 'if' block
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(tfile.name)
        
        # Load the YOLO model
        model = load_yolo_model()

        # Create a placeholder to display the video feed
        video_placeholder = st.empty()
        
        # Initialize a counter for total detected vehicles
        total_vehicles = 0
        
        st.subheader("Object Detection in Progress...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 inference on the frame
            results = model(frame)
            
            # Count the number of detected vehicles (YOLO class '2' for car, '3' for motorcycle, '5' for bus, '7' for truck)
            vehicle_classes = [2, 3, 5, 7]
            detections = [det for det in results[0].boxes.cls.tolist() if int(det) in vehicle_classes]
            current_vehicles = len(detections)
            total_vehicles += current_vehicles
            
            # Annotate the frame with bounding boxes and labels
            annotated_frame = results[0].plot()

            # Convert frame to RGB for Streamlit display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the annotated frame in the placeholder
            video_placeholder.image(annotated_frame_rgb, use_column_width=True)
            
            # Small delay to simulate real-time processing
            time.sleep(0.01)
            
        cap.release()
        st.success("Analysis complete!")

# --- Column 2: Key Metrics and Data ---
with col2:
    st.header("Real-Time Metrics")
    
    # Placeholders for dynamic metrics
    metric1 = st.metric(label="Vehicles in Frame", value=0)
    metric2 = st.metric(label="Total Vehicles Detected", value=0)
    metric3 = st.metric(label="Average Wait Time", value="N/A")

    # This part would typically be updated within your video processing loop
    # or by an external process connected to a live SUMO simulation.
    # For this example, we'll update the metrics based on the video
    # analysis results for demonstration purposes.
    if video_file:
        # These lines must also be indented to be part of the 'if' block
        st.metric(label="Vehicles in Frame", value=current_vehicles)
        st.metric(label="Total Vehicles Detected", value=total_vehicles)
        
    st.header("Simulation Controls")
    st.markdown("""
    _This section would contain controls for the SUMO simulation._
    
    You would use the `traci` library to start and control the simulation from Python.
    """)
    if st.button("Start SUMO Simulation"):
        st.info("Simulation started! (Placeholder - requires SUMO setup)")
        # Example:
        # import traci
        # traci.start([...])
        # Your simulation loop would go here.
        
    if st.button("Stop Simulation"):
        st.info("Simulation stopped! (Placeholder)")
        # Example:
        # traci.close()

st.sidebar.markdown("### About ATRICS")
st.sidebar.info("ATRICS is a project for the Smart India Hackathon 2025 focusing on an AI-based traffic regulation system. It uses YOLOv8 for object detection and SUMO for traffic simulation.")