import cv2
import mediapipe as mp
import os
import pandas as pd

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

landmark_names = [
    "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer",
    "Right Eye Inner", "Right Eye", "Right Eye Outer",
    "Left Ear", "Right Ear", "Mouth Left", "Mouth Right",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky",
    "Left Index", "Right Index", "Left Thumb", "Right Thumb",
    "Left Hip", "Right Hip", "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle", "Left Heel", "Right Heel",
    "Left Foot Index", "Right Foot Index"
]

num_landmarks = len(landmark_names)
print(f"Number of landmarks: {num_landmarks}")


camera_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos'
video_name = 'neg_infer'
video_path = camera_path +'/' + video_name +'.mp4'
save_path = camera_path + '/labeled_data/' + video_name
# output_csv = camera_path + '/video/landmark/pose.csv'

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
csv_data = []
results = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame

    row = [video_name]
    # pose_landmarks >> pose_world_landmarks : absolute coordinate system : 23 <-> 24
    if result.pose_world_landmarks:
        for landmark in result.pose_world_landmarks.landmark:
            x = landmark.x  # 0~1 범위의 x 좌표
            y = landmark.y  # 0~1 범위의 y 좌표
            z = landmark.z  # 3D 공간에서의 z 좌표
            row.extend([x, y, z])  # x, y, z 좌표 추가
    else:
        # 랜드마크가 없을 경우 NaN 추가
        row.extend([None] * (num_landmarks * 3))  # x, y, z 좌표 각각 NaN 추가
    results.append(row)
    # filename = save_path+'/'+str(frame_number)+'.jpg'

    # Display the frame
    # cv2.imwrite(filename, frame)
# 결과를 DataFrame으로 변환
if results:
    # 모든 행이 동일한 길이를 갖도록 NaN으로 채운다
    max_length = max(len(res) for res in results)
    for res in results:
        while len(res) < max_length:
            res.append(None)

    # 컬럼 생성: 파일 이름 + 각 랜드마크의 x, y, z 좌표
    columns = ['filename'] + [f'{name}_{axis}' for name in landmark_names for axis in ['x', 'y', 'z']]
    df = pd.DataFrame(results, columns=columns)

    # CSV 파일을 저장할 경로 설정
    os.makedirs(save_path, exist_ok=True)

    # CSV 파일명 설정
    output_csv = os.path.join(save_path, f"landmarks_3d_{video_name}.csv")

    # CSV 파일로 저장 (덮어쓰기)
    df.to_csv(output_csv, index=False)

    print(f"3D 랜드마크 좌표가 {output_csv} 파일에 저장되었습니다.")
