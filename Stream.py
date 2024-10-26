import streamlit as st
import cv2
import mediapipe as mp
from fer import FER
import requests
import time
import os
import pandas as pd
import folium
from streamlit_folium import folium_static
from huggingface_hub import InferenceClient
import speech_recognition as sr
from gtts import gTTS
import plotly.express as px

# Initialize MediaPipe and FER components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
emotion_detector = FER()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Streamlit configuration
st.set_page_config(page_title="Ladli - Women's Safety Platform", layout="wide")

# Initialize session state
if 'emergency_data' not in st.session_state:
    st.session_state.emergency_data = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "How may I assist you with women's safety today?"}]

def main():
    # Sidebar navigation
    st.sidebar.title("Ladli Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Cover Page", "Alert Panel", "Risk Areas", "Services", "Emergency Detection"]
    )

    if page == "Cover Page":
        show_cover_page()
    elif page == "Alert Panel":
        show_alert_panel()
    elif page == "Risk Areas":
        show_risk_areas()
    elif page == "Services":
        show_services()
    elif page == "Emergency Detection":
        run_emergency_detection()

def show_cover_page():
    st.title("🛡️ Welcome to Ladli - Women's Safety Platform")
    
    st.markdown("""### Empowering Women Through Safety and Technology
    Ladli is a comprehensive women's safety platform that combines real-time emergency detection,
    location tracking, and advisory services to ensure women's safety across India.
    
     Key Features:
    - 👁️ Real-time emergency gesture detection
    - 📍 Location tracking and risk area mapping
    - 🚨 Immediate alert system
    - 🤖 AI-powered safety advisory
    - 📱 Emergency services connection
    """)

    # Self-defense techniques section
    st.subheader("Essential Self-Defense Techniques")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        1. **Basic Stance**
           - Keep feet shoulder-width apart
           - Maintain balance
           - Stay alert and aware
           
        2. **Strike Points**
           - Eyes
           - Throat
           - Groin
           - Knees
        """)
    
    with col2:
        st.markdown("""
        3. **Emergency Moves**
           - Palm strike
           - Knee strike
           - Elbow strike
           - Quick escape techniques
           
        4. **Safety Tips**
           - Trust your instincts
           - Stay aware of surroundings
           - Keep emergency contacts ready
        """)

def show_alert_panel():
    st.title("🚨 Alert Panel")
    
    if st.session_state.emergency_data:
        for alert in st.session_state.emergency_data:
            with st.expander(f"Emergency Alert - {alert['timestamp']}"):
                st.write(f"**Emergency Score:** {alert['score']}%")
                st.write(f"**Location:** {alert['city']}, {alert['region']}, {alert['country']}")
                st.write(f"**People Present:** {alert['num_people']}")
                st.write(f"**Google Maps Link:** {alert['maps_link']}")
                
                if 'snapshot_path' in alert:
                    st.image(alert['snapshot_path'], caption="Emergency Snapshot")
    else:
        st.info("No emergency alerts recorded yet.")

def show_risk_areas():
    st.title("🗺️ Risk Areas Analysis")
    
    # Sample risk data (replace with real data in production)
    risk_data = pd.DataFrame({
        'City': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
        'Latitude': [19.0760, 28.6139, 12.9716, 13.0827, 22.5726],
        'Longitude': [72.8777, 77.2090, 77.5946, 80.2707, 88.3639],
        'Risk_Level': [75, 85, 60, 65, 70]
    })
    
    # Create map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
    
    for idx, row in risk_data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Risk_Level']/5,
            popup=f"{row['City']}: Risk Level {row['Risk_Level']}%",
            color='red',
            fill=True
        ).add_to(m)
    
    # Display map
    st.subheader("Women's Safety Risk Heat Map")
    folium_static(m)
    
    # Risk statistics
    st.subheader("Risk Analysis")
    fig = px.bar(risk_data, x='City', y='Risk_Level',
                 title='Safety Risk Levels by City')
    st.plotly_chart(fig)

def show_services():
    st.title("📞 Emergency Services")
    
    st.markdown("""
    ### Important Emergency Numbers
    
    #### National Emergency Numbers
    - **Women's Helpline:** 1091
    - **Police:** 100
    - **Ambulance:** 102
    - **Fire:** 101
    - **Emergency Management Services:** 108
    
    #### Women's Safety Apps
    - Ladli Emergency App
    - Government Women Safety Portal
    - Local Police App
    
    #### Support Organizations
    - National Commission for Women
    - State Women's Commissions
    - Local Women's Help Groups
    """)

# def calculate_gesture_score(hand_landmarks):
#     # Your existing implementation here...
#     gesture_score = 0
#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             gesture_score = calculate_gesture_score(hand_landmarks)

def get_location_details():
    response = requests.get("https://ipinfo.io/json")
    if response.ok:
        return response.json()
    return None

def run_emergency_detection():
    st.title("👁️ Emergency Detection System")
    
    if st.button("Start Emergency Detection", key="start_detection_button"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        last_snapshot_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get location details
            location_details = get_location_details()
            if location_details:
                ip = location_details.get("ip")
                city = location_details.get("city")
                region = location_details.get("region")
                country = location_details.get("country")
                loc = location_details.get("loc")
                latitude, longitude = loc.split(",") if loc else (None, None)
                google_maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
            else:
                ip, city, region, country = [None] * 4
                google_maps_link = "Location not available"

            # Convert image to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand gestures
            hand_results = hands.process(rgb_frame)  # Ensure hand_results is defined here
            gesture_score = 0
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture_score = calculate_gesture_score(hand_landmarks)  # Now passing hand_landmarks

            # Check for emergency condition
            if gesture_score == 80:
                emergency_message = "Emergency Situation Detected!"
                cv2.putText(frame, emergency_message, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Store emergency data
                alert_data = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'score': gesture_score,
                    'city': city,
                    'region': region,
                    'country': country,
                    'maps_link': google_maps_link,
                    'snapshot_path': f'snapshots/snapshot_{int(time.time())}.jpg'  # Placeholder for the snapshot
                }
                st.session_state.emergency_data.append(alert_data)

                # Save snapshot
                cv2.imwrite(alert_data['snapshot_path'], frame)
                print("Emergency detected and data saved.")

                break  # Break after detection

            # Display score
            cv2.putText(frame, f"Emergency Score: {gesture_score}%", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show processed frame
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Capture snapshot every 2 seconds
            current_time = time.time()
            if current_time - last_snapshot_time >= 2:
                snapshot_path = f'snapshots/snapshot_{int(current_time)}.jpg'
                cv2.imwrite(snapshot_path, frame)
                last_snapshot_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
