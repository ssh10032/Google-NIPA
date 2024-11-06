import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import pandas as pd

# Initialize MediaPipe PoseLandmarker using the latest API
base_options = python.BaseOptions(model_asset_path='/home/baebro/nipa_ws/nipaproj_ws/landmark_labeling/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Define landmark names
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

# Set image folder path
image_folder = "/media/baebro/NIPA_data/Train"
# image_folder = "/home/baebro/nipa_ws/nipaproj_ws/sample_videos/anchor"
output_base_folder = os.path.join(image_folder, "landmarks_recent")

# Process images in the folder
for root, dirs, files in os.walk(image_folder):
    if root.endswith('_F'):
        continue
    image_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if image_files:
        parent_folder_name = os.path.basename(os.path.dirname(root))
        base_folder_name = parent_folder_name.replace("[원천]", "").replace(".zip_unzipped", "").strip()
        subfolder_last_char = os.path.basename(root)[-1]
        results = []

        for filename in image_files:
            image_path = os.path.join(root, filename)

            try:
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Run pose estimation
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = pose_landmarker.detect(mp_image)

                # Extract landmark coordinates
                row = [filename]
                if detection_result.pose_world_landmarks:
                    for landmark in detection_result.pose_world_landmarks[0]:
                        row.extend([landmark.x, landmark.y, landmark.z])
                else:
                    row.extend([None] * (len(landmark_names) * 3))

                results.append(row)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

        if results:
            max_length = max(len(res) for res in results)
            for res in results:
                while len(res) < max_length:
                    res.append(None)

            columns = ['filename'] + [f'{name}_{axis}' for name in landmark_names for axis in ['x', 'y', 'z']]
            df = pd.DataFrame(results, columns=columns)

            output_subfolder = os.path.join(output_base_folder, base_folder_name)
            os.makedirs(output_subfolder, exist_ok=True)
            output_csv = os.path.join(output_subfolder, f"landmarks_3d_{subfolder_last_char}.csv")
            df.to_csv(output_csv, index=False)

            print(f"3D landmark coordinates saved to {output_csv}")
