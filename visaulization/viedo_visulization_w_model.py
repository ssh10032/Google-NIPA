import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from model.model import skeletonLSTM
import imageio


def read_angles(file_path, sequence_length):
    angle_list = []
    df = pd.read_csv(file_path)
    for i in range(0, 300, sequence_length):
        angle_list.append(torch.tensor(df.iloc[i:i + sequence_length].values, dtype=torch.float32))
    pos1_tensor = torch.stack(angle_list, dim=0).to(device)
    return pos1_tensor


def calc_score(tensor1, tensor2, model):
    emb1 = model(tensor1)
    emb2 = model(tensor2)

    emb1_norm = F.normalize(emb1, dim=1)
    emb2_norm = F.normalize(emb2, dim=1)
    return 1 - torch.pow(F.pairwise_distance(emb1_norm, emb2_norm), 1) / 2


saved_path = '/home/baebro/nipa_ws/nipaproj_ws/output/pt files/'
model_name = 'model_state_dict_lstm_2000_add_ov5.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model load
model = skeletonLSTM(12, 64)
model.load_state_dict(torch.load(saved_path + model_name))
model.to(device)

# read angle csv
origin_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/labeled_data/pos1_infer/landmarks_3d_pos1_infer_angles.csv'
pos_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/labeled_data/pos2_infer/landmarks_3d_pos2_infer_angles.csv'
neg_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/labeled_data/neg_infer/landmarks_3d_neg_infer_angles.csv'

pos1_tensor = read_angles(origin_path, 30).to(device)
# pos_input = read_angles(pos_path, 30).to(device)
pos2_tensor = read_angles(pos_path, 30).to(device)

# 동영상 파일 경로
video_path_1 = "/home/baebro/nipa_ws/nipaproj_ws/sample_videos/pos1_infer.mp4"  # 첫 번째 동영상
video_path_2 = "/home/baebro/nipa_ws/nipaproj_ws/sample_videos/pos2_infer.mp4"  # 첫 번째 동영상
# video_path_2 = "/home/baebro/nipa_ws/nipaproj_ws/sample_videos/neg.mp4"  # 두 번째 동영상

# 동영상 파일 열기
cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

idx = 0
cnt = 0
text = ''
idx = 0
cnt = 0

frames = []  # List to store frames for GIF creation

# 동영상이 열렸는지 확인
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one of the videos.")
else:
    while True:
        cnt += 1
        ret1, frame1 = cap1.read()  # 첫 번째 동영상 프레임 읽기
        ret2, frame2 = cap2.read()  # 두 번째 동영상 프레임 읽기

        if not ret1 or not ret2:
            print("End of video or error while reading the videos.")
            break

        if cnt == 30:
            input1 = pos1_tensor[idx].unsqueeze(0)
            input2 = pos2_tensor[idx].unsqueeze(0)

            score = calc_score(input1, input2, model)
            text = f"{int(score.item() * 100)}"

            cnt = 0
            idx += 1

        # 두 프레임의 크기를 동일하게 맞추기
        height = max(frame1.shape[0], frame2.shape[0]) // 2
        width = max(frame1.shape[1], frame2.shape[1]) // 2

        # 두 프레임을 같은 높이로 맞추기
        frame1_resized = cv2.resize(frame1, (width, height))
        frame2_resized = cv2.resize(frame2, (width, height))

        space_width = 100
        space_height = 75
        white_space1 = np.ones((height, space_width, 3), dtype=np.uint8) * 255

        total_height = height + space_height
        total_width = width * 2 + space_width

        # 두 프레임을 수평으로 결합
        combined_frame = np.hstack((frame1_resized, white_space1, frame2_resized))

        white_space2 = np.ones((space_height, space_width + width * 2, 3), dtype=np.uint8) * 255

        combined_frame = np.vstack((white_space2, combined_frame))

        font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 설정
        font_scale = 1  # 폰트 크기
        font_thickness = 2  # 두께
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]  # 텍스트 크기 계산

        text_x = (total_width - text_size[0]) // 2
        text_y = (total_height + text_size[1]) // 2

        cv2.putText(combined_frame, text, (text_x, text_y),
                    font, font_scale, (50, 50, 50),
                    font_thickness, cv2.LINE_AA)
        # 결합된 프레임을 화면에 표시
        cv2.imshow('Combined Video Playback', combined_frame)

        # Store the frame for GIF creation
        frames.append(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))

        # 'q' 키를 눌러 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Save the frames as a GIF
gif_path = '/home/baebro/nipa_ws/nipaproj_ws/output/combined_video.gif'
imageio.mimsave(gif_path, frames, fps=10)

# 리소스 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
