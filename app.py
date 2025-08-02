import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:\\Users\\manoj\\smart safety gear detection system\\runs\\detect\\yolov8n_v1_train10\\weights\\best.pt")

# Streamlit UI
st.title("Smart Safety Gear Detection")
st.write("Upload an image or video to detect safety gear.")

# Detection mode selection using radio buttons with icons
detection_mode = st.radio(
    "Select Detection Mode",
    options=["🖼️ Image Detection", "🎥 Video Detection"]
)

# Initialize a session state to track if video processing is completed
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False  # Set initial state to False

# Safety gear classes to check (positive and negative)
safety_gear_classes = ['Hardhat', 'Mask', 'Safety Vest', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

# File uploader
if detection_mode == "🖼️ Image Detection":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated to use_container_width
        st.write("Detecting...")

        # Perform inference
        results = model(image)

        # Convert to annotated image
        annotated_image = results[0].plot()

        # Ensure color format is RGB
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Display annotated image in Streamlit
        st.image(annotated_image_rgb, caption='Detected Image', use_container_width=True)  # Updated to use_container_width

        # Get detected classes (detected gear)
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        
        # Logic to determine missing safety gear
        missing_classes = []
        for gear in safety_gear_classes:
            if gear.startswith('NO-'):
                positive_class = gear[3:]  # Remove "NO-" from class name
                if positive_class not in detected_classes:
                    missing_classes.append(gear)
            else:
                negative_class = 'NO-' + gear  # Corresponding negative class (e.g., NO-Hardhat)
                if gear not in detected_classes and negative_class not in detected_classes:
                    missing_classes.append(gear)

        # Show missing classes warning
        if missing_classes:
            st.warning(f"⚠️ Missing Safety Gear: {', '.join(missing_classes)}", icon="⚠️")

elif detection_mode == "🎥 Video Detection":
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None and not st.session_state.video_processed:
        # Save uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video.close()

        # Open the input video with OpenCV
        cap = cv2.VideoCapture(temp_video.name)
        output_path = "detected_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None  # Initialize video writer variable

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)  # Initialize the progress bar

        st.write("Detecting on video...")

        # Process video frame by frame
        missing_classes_set = set()  # Track missing classes throughout the video
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on each frame
            results = model(frame)
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
            annotated_frame = results[0].plot()  # Annotate the frame

            # Logic to determine missing safety gear for each frame
            for gear in safety_gear_classes:
                if gear.startswith('NO-'):
                    positive_class = gear[3:]  # Remove "NO-" from class name
                    if positive_class not in detected_classes:
                        missing_classes_set.add(gear)
                else:
                    negative_class = 'NO-' + gear  # Corresponding negative class (e.g., NO-Hardhat)
                    if gear not in detected_classes and negative_class not in detected_classes:
                        missing_classes_set.add(gear)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Update the progress bar
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

        # Release resources
        cap.release()
        out.release()
        os.remove(temp_video.name)

        # Display the annotated video
        st.video(output_path)
        st.write("Video processing completed.")

        # Check for missing classes and notify using a warning
        if missing_classes_set:
            missing_classes = list(missing_classes_set)
            st.warning(f"⚠️ Missing Safety Gear: {', '.join(missing_classes)}", icon="⚠️")

        # Ensure the output video exists before providing a download link
        if os.path.exists(output_path):
            # Update session state to indicate processing is complete
            st.session_state.video_processed = True

        # Display the download button only if processing is complete
        if st.session_state.video_processed:
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="detected_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Video processing failed. Please try again.")
