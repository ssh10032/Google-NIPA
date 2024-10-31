import cv2
import mediapipe as mp
import csv

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    # print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        # print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
        # print(idx)
    # print(mp_pose.PoseLandmark(23).name)
    # print(mp_pose.PoseLandmark(24).name)
    # print('x coord is ',(landmarks[23].x+landmarks[24].x)/2)
    # print('y coord is ', (landmarks[23].y+landmarks[24].y)/2)
    # print('z coord is ', (landmarks[23].z+landmarks[24].z)/2)
    print("\n")

camera_path = '/home/baebro/nipa_ws/nipaproj_ws'
video_path = camera_path + '/neg.mp4'
save_path = camera_path + '/visaulization/vis_output/neg'
output_csv = camera_path + '/video/landmark/pose.csv'

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
csv_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame


    # pose_landmarks >> pose_world_landmarks : absolute coordinate system : 23 <-> 24
    if result.pose_world_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # mp_drawing.draw_landmarks(frame, result.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(result.pose_world_landmarks.landmark, frame_number, csv_data)
        frame_number += 1

    filename = save_path+'/'+str(frame_number)+'.jpg'

    # Display the frame
    cv2.imwrite(filename, frame)
