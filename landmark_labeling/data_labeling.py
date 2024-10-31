import cv2
import mediapipe as mp
import os
import pandas as pd

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# 랜드마크 이름 목록
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

# 랜드마크 개수 확인
num_landmarks = len(landmark_names)
print(f"Number of landmarks: {num_landmarks}")

# 이미지 폴더 경로 설정
image_folder = "/media/baebro/NIPA_data/Train"  # 여기에 최상위 폴더 경로를 입력하세요
output_base_folder = os.path.join(image_folder, "landmarks")

# 하위 폴더 포함 모든 이미지 파일에 대해 처리
for root, dirs, files in os.walk(image_folder):
    # 이미지 파일 필터링 및 정렬
    image_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # 이미지 파일이 있는 경우에만 처리
    if image_files:
        # 최상위 폴더 이름에서 "[원천]"과 ".zip_unzipped"를 제거
        parent_folder_name = os.path.basename(os.path.dirname(root))
        base_folder_name = parent_folder_name.replace("[원천]", "").replace(".zip_unzipped", "").strip()

        # 현재 폴더의 마지막 문자 추출
        subfolder_last_char = os.path.basename(root)[-1]

        # 이미지 파일 이름을 정렬하여 시계열 순서로 처리
        results = []

        for filename in image_files:
            image_path = os.path.join(root, filename)

            try:
                # 이미지 읽기
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 포즈 감지
                result = pose.process(image_rgb)

                # 랜드마크 좌표 추출
                row = [filename]  # 파일 이름
                if result.pose_world_landmarks:
                    for landmark in result.pose_world_landmarks.landmark:
                        x = landmark.x  # 0~1 범위의 x 좌표
                        y = landmark.y  # 0~1 범위의 y 좌표
                        z = landmark.z  # 3D 공간에서의 z 좌표
                        row.extend([x, y, z])  # x, y, z 좌표 추가
                else:
                    # 랜드마크가 없을 경우 NaN 추가
                    row.extend([None] * (num_landmarks * 3))  # x, y, z 좌표 각각 NaN 추가

                # 결과 리스트에 추가
                results.append(row)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

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
            output_subfolder = os.path.join(output_base_folder, base_folder_name)
            os.makedirs(output_subfolder, exist_ok=True)

            # CSV 파일명 설정
            output_csv = os.path.join(output_subfolder, f"landmarks_3d_{subfolder_last_char}.csv")

            # CSV 파일로 저장 (덮어쓰기)
            df.to_csv(output_csv, index=False)

            print(f"3D 랜드마크 좌표가 {output_csv} 파일에 저장되었습니다.")
