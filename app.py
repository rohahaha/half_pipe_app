import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from collections import deque
import time
import plotly.graph_objs as go

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

def process_video(video_file, tracker_type, bbox):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_filename = tmp_file.name

    video = cv2.VideoCapture(temp_filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = video.read()
    if not ret:
        st.error("Failed to read the video")
        return

    tracker = create_tracker(tracker_type)
    tracker.init(first_frame, bbox)

    prev_pos = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
    pixels_per_meter = 30  # This value should be calibrated for accurate speed
    speed_queue = deque(maxlen=5)  # Store last 5 speed measurements

    stframe = st.empty()
    speed_text = st.empty()
    
    speeds = []
    frames = []
    
    graph_placeholder = st.empty()

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

            curr_pos = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
            speed = calculate_speed(prev_pos, curr_pos, fps, pixels_per_meter)
            speed_queue.append(speed)
            avg_speed = sum(speed_queue) / len(speed_queue)

            cv2.putText(frame, f"Speed: {avg_speed:.2f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            prev_pos = curr_pos
            speed_text.text(f"실시간 속력: {avg_speed:.2f} km/h")
            
            speeds.append(avg_speed)
            frames.append(frame_count)
        else:
            cv2.putText(frame, "Tracking failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR", use_column_width=True)
        
        fig = go.Figure(data=go.Scatter(x=frames, y=speeds, mode='lines', line=dict(color=st.session_state.graph_color)))
        fig.update_layout(title="Speed over time",
                          xaxis_title="Frame",
                          yaxis_title="Speed (km/h)",
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')
        graph_placeholder.plotly_chart(fig, use_container_width=True)
        
        frame_count += 1
        time.sleep(1/fps)  # Simulate real-time playback

    video.release()
    os.unlink(temp_filename)

st.title('움직이는 물체 실시간 속도 측정기 :-) (ROHA_240831)')

uploaded_file = st.file_uploader("비디오 파일을 선택하세요.", type=['mp4', 'avi', 'mov'])

if 'graph_color' not in st.session_state:
    st.session_state.graph_color = 'white'

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    st.video(video_bytes)
    
    tracker_type = st.selectbox(
        '속도 트래커를 선택해주세요. (선생님께 문의하기!)',
        ('MIL', 'KCF', 'CSRT', 'MOSSE')
    )
    
    selection_method = st.radio(
        '추적 대상 선택 방법:',
        ('Point', 'Box')
    )

    # 비디오의 첫 프레임 가져오기
    video = cv2.VideoCapture(uploaded_file.name)
    ret, first_frame = video.read()
    video.release()

    if ret:
        height, width = first_frame.shape[:2]
        
        if selection_method == 'Point':
            col1, col2 = st.columns(2)
            with col1:
                x = st.slider('X 좌표', 0, width, width // 2)
            with col2:
                y = st.slider('Y 좌표', 0, height, height // 2)
            
            # 점 주변에 작은 박스 생성
            box_size = 20
            bbox = (x - box_size//2, y - box_size//2, box_size, box_size)
        else:  # Box
            col1, col2 = st.columns(2)
            with col1:
                x = st.slider('X 좌표', 0, width, width // 4)
                w = st.slider('너비', 10, width - x, width // 2)
            with col2:
                y = st.slider('Y 좌표', 0, height, height // 4)
                h = st.slider('높이', 10, height - y, height // 2)
            
            bbox = (x, y, w, h)
        
        # 선택한 영역을 첫 프레임에 표시
        frame_with_selection = first_frame.copy()
        cv2.rectangle(frame_with_selection, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        st.image(frame_with_selection, channels="BGR", use_column_width=True)
        
        st.session_state.graph_color = st.radio(
            "그래프 색상을 선택해주세요. (처음엔 white로 선택하세요!)",
            ('white', 'black')
        )
        
        if st.button('영상 내 속도 추적 시작하기'):
            st.write("Processing video...")
            process_video(uploaded_file, tracker_type, bbox)
    else:
        st.error("Failed to read the first frame of the video.")
