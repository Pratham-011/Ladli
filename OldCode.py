import cv2
import mediapipe as mp
from fer import FER  # Facial Emotion Recognition library
import requests  # To get IP geolocation
import time
import os

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize emotion detection
emotion_detector = FER()

# Initialize face detection for person counting
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set up the video recording
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('recorded_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

# Ensure the snapshot directory exists
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# Function to calculate gesture score based on similarity to the ASL "U" sign
def calculate_gesture_score(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    index_middle_up = (index_tip.y < index_dip.y and middle_tip.y < middle_dip.y and 
                       abs(index_tip.x - middle_tip.x) < 0.05)
    ring_folded = ring_tip.y > index_dip.y
    pinky_folded = pinky_tip.y > index_dip.y
    thumb_position = thumb_tip.y > index_dip.y

    score = 0
    if index_middle_up:
        score += 30  # Accurate positioning of index and middle fingers
    if ring_folded:
        score += 20  # Ring finger folded
    if pinky_folded:
        score += 20  # Pinky finger folded
    if thumb_position:
        score += 10  # Thumb positioned correctly

    return min(score, 80)  # Cap gesture score at 80

# Function to get current location details
def get_location_details():
    response = requests.get("https://ipinfo.io/json")  # Get location based on IP
    if response.ok:
        return response.json()
    return None

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
        loc = location_details.get("loc")  # Latitude,Longitude
        org = location_details.get("org")
        postal = location_details.get("postal")
        timezone = location_details.get("timezone")

        latitude, longitude = loc.split(",") if loc else (None, None)
        google_maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
    else:
        ip, city, region, country, loc, org, postal, timezone = [None] * 8
        google_maps_link = "Location not available"

    # Convert image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand gestures
    hand_results = hands.process(rgb_frame)

    # Detect faces for person count
    face_results = face_detection.process(rgb_frame)
    num_people = len(face_results.detections) if face_results.detections else 0

    # Calculate gesture score based on ASL "U" gesture similarity
    gesture_score = 0
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_score = calculate_gesture_score(hand_landmarks)

    # Display emergency message and stop if score equals 80
    if gesture_score == 80:
        emergency_message = "Emergency Situation Detected!"
        cv2.putText(frame, emergency_message, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(emergency_message)
        print(f"Emergency Score: {gesture_score}%")
        print(f"IP Address: {ip}")
        print(f"City: {city}")
        print(f"Region: {region}")
        print(f"Country: {country}")
        print(f"Location: (Latitude: {latitude}, Longitude: {longitude})")
        print(f"Organization: {org}")
        print(f"Postal Code: {postal}")
        print(f"Timezone: {timezone}")
        print(f"Google Maps Link: {google_maps_link}")
        print(f"Number of People Detected: {num_people}")
        break
    else:
        cv2.putText(frame, f"Emergency Score: {gesture_score}%", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Emergency Score: {gesture_score}%")
        print(f"IP Address: {ip}")
        print(f"City: {city}")
        print(f"Region: {region}")
        print(f"Country: {country}")
        print(f"Location: (Latitude: {latitude}, Longitude: {longitude})")
        print(f"Organization: {org}")
        print(f"Postal Code: {postal}")
        print(f"Timezone: {timezone}")
        print(f"Google Maps Link: {google_maps_link}")
        print(f"Number of People Detected: {num_people}")

    # Display number of people detected on the frame
    cv2.putText(frame, f"People Count: {num_people}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Write frame to the video file
    out.write(frame)

    # Capture snapshot every 2 seconds
    current_time = time.time()
    if current_time - last_snapshot_time >= 2:
        snapshot_path = f'snapshots/snapshot_{int(current_time)}.jpg'
        cv2.imwrite(snapshot_path, frame)
        print(f"Snapshot saved at: {snapshot_path}")
        last_snapshot_time = current_time

    # Show the processed frame
    cv2.imshow("Emergency Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
