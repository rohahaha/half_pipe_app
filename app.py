import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from collections import deque
import time

def get_first_frame(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_filename = tmp_file.name

    video = cv2.VideoCapture(temp_filename)
    ret, frame = video.read()
    video.release()
    os.unlink(temp_filename)
    return frame if ret else None

def calculate_speed(prev_pos, curr_pos, fps, pixels_per_meter):
    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    distance_meters = distance / pixels_per_meter
    speed = distance_meters * fps * 3.6  # Convert m/s to km/h
    return speed

def create_tracker(tracker_type):
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    else:
        st.error(f"Unsupported tracker type: {tracker_type}")
        return None

def process_video(video_file, bbox, tracker_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_filename = tmp_file.name

    video = cv2.VideoCapture(temp_filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = create_tracker(tracker_type)
    if tracker is None:
        return

    ret, frame = video.read()
    if not ret:
        st.error("Failed to read the video")
        return

    tracker.init(frame, bbox)
    prev_pos = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
    pixels_per_meter = 30  # This value should be calibrated for accurate speed
    speed_queue = deque(maxlen=5)  # Store last 5 speed measurements

    stframe = st.empty()
    speed_text = st.empty()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

            curr_pos = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            speed = calculate_speed(prev_pos, curr_pos, fps, pixels_per_meter)
            speed_queue.append(speed)
            avg_speed = sum(speed_queue) / len(speed_queue)

            cv2.putText(frame, f"Speed: {avg_speed:.2f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            prev_pos = curr_pos
            speed_text.text(f"Current Speed: {avg_speed:.2f} km/h")
        else:
            cv2.putText(frame, "Tracking failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR", use_column_width=True)
        time.sleep(1/fps)  # Simulate real-time playback

    video.release()
    os.unlink(temp_filename)

st.title('Object Tracking and Speed Estimation')

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    first_frame = get_first_frame(uploaded_file)
    if first_frame is not None:
        st.write("Select the object to track:")
        height, width = first_frame.shape[:2]
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider('X coordinate', 0, width, width // 4)
            y = st.slider('Y coordinate', 0, height, height // 4)
        with col2:
            w = st.slider('Width', 10, width - x, width // 4)
            h = st.slider('Height', 10, height - y, height // 4)
        
        bbox = (x, y, w, h)
        
        # Draw rectangle on first frame
        cv2.rectangle(first_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(first_frame, channels="BGR", use_column_width=True)
        
        tracker_type = st.selectbox(
            'Select a tracker',
            ('CSRT', 'KCF', 'MOSSE', 'MIL')
        )
        
        if st.button('Process Video'):
            st.write("Processing video...")
            uploaded_file.seek(0)
            process_video(uploaded_file, bbox, tracker_type)
            st.write("Video processing completed.")
    else:
        st.error("Failed to read the first frame of the video.")

st.write("OpenCV version:", cv2.__version__)