import pandas as pd
import numpy as np
import os

# 파일 경로 및 데이터 설정
input_dir = '/media/baebro/NIPA_data/Train/landmarks'
input_dir = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/labeled_data'

angle_pairs = [
    ['left_biceps', 'left_forearm'],
    ['right_biceps', 'right_forearm'],
    ['between_shoulders', 'left_body'],
    ['between_shoulders', 'right_body'],
    ['between_shoulders', 'right_neck'],
    ['between_shoulders', 'left_neck'],
    ['between_pelvis', 'left_thigh'],
    ['between_pelvis', 'right_thigh'],
    ['right_thigh', 'right_calf'],
    ['left_thigh', 'left_calf'],
    ['right_body', 'right_thigh'],
    ['left_body', 'left_thigh']
]

body_parts = {
    'left_biceps': [11, 13],
    'left_forearm': [13, 15],
    'right_biceps': [12, 14],
    'right_forearm': [14, 16],
    'between_shoulders': [11, 12],
    'left_body': [11, 23],
    'right_body': [12, 24],
    'between_pelvis': [23, 24],
    'left_thigh': [23, 25],
    'left_calf': [25, 27],
    'right_thigh': [24, 26],
    'right_calf': [26, 28],
    'left_neck': [9, 11],
    'right_neck': [10, 12]
}

# 각도 계산 함수
class AngleCalculator:
    def __init__(self, angle_pairs, body_parts):
        self.angle_pairs = angle_pairs
        self.body_parts = body_parts

    def calculate_angles(self, df):
        angle_data = []

        for part1, part2 in self.angle_pairs:
            vec1 = df.iloc[:, self.body_parts[part1][0] * 3 + 1:self.body_parts[part1][0] * 3 + 4].values - \
                   df.iloc[:, self.body_parts[part1][1] * 3 + 1:self.body_parts[part1][1] * 3 + 4].values
            vec2 = df.iloc[:, self.body_parts[part2][0] * 3 + 1:self.body_parts[part2][0] * 3 + 4].values - \
                   df.iloc[:, self.body_parts[part2][1] * 3 + 1:self.body_parts[part2][1] * 3 + 4].values
            dot_product = np.einsum('ij,ij->i', vec1, vec2)
            norms = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
            angles = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
            angle_data.append(angles)

        return np.stack(angle_data, axis=1)

# 디렉토리 탐색 및 각도 계산
angle_calculator = AngleCalculator(angle_pairs, body_parts)

for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.csv'):
            input_csv = os.path.join(root, file)
            df = pd.read_csv(input_csv)

            # 각도 계산
            angles = angle_calculator.calculate_angles(df)

            # 각도 데이터를 데이터프레임으로 변환 후 하나의 CSV로 저장
            angle_columns = [f'{part1}_{part2}_angle' for part1, part2 in angle_pairs]
            angle_df = pd.DataFrame(angles, columns=angle_columns)
            output_filename = f'{os.path.splitext(file)[0]}_angles.csv'
            output_path = os.path.join(root, output_filename)
            angle_df.to_csv(output_path, index=False)

print("모든 디렉토리의 각도 계산 및 저장 완료!")
