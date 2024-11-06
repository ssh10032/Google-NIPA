from model import skeletonLSTM, head
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

def read_angle(file_path):
    df = pd.read_csv(file_path)
    segment = df.iloc[:30].values
    segment = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(torch.float32)
    return segment

def read_angles(file_path, sequence_length):
    angle_list = []
    df = pd.read_csv(file_path)
    for i in range(0, 300, sequence_length):
        angle_list.append(torch.tensor(df.iloc[i:i + sequence_length].values, dtype=torch.float32))
    pos1_tensor = torch.stack(angle_list, dim=0).to(device)
    return pos1_tensor

# origin_path = '/media/baebro/NIPA_data/Train/landmarks/빨간맛5 (레드벨벳)/landmarks_3d_L_angles.csv'
# pos_path = '/media/baebro/NIPA_data/Train/landmarks/빨간맛5 (레드벨벳)/landmarks_3d_R_angles.csv'
# neg_path = '/media/baebro/NIPA_data/Train/landmarks/소원을 말해봐 (소녀시대)/landmarks_3d_L_angles.csv'

origin_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/anchor/labeled_data_full/pos1_infer/landmarks_3d_pos1_infer_angles.csv'
pos_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/anchor/labeled_data_full/pos2_infer/landmarks_3d_pos2_infer_angles.csv'
neg_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/anchor/labeled_data_full/neg_infer/landmarks_3d_neg_infer_angles.csv'

# origin_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/mara/labeled_data_full/pos_1/landmarks_3d_pos_1_angles.csv'
# pos_path = '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/mara/labeled_data_full/pos_2/landmarks_3d_pos_2_angles.csv'
# neg_path ='/home/baebro/nipa_ws/nipaproj_ws/sample_videos/mara/labeled_data_full/neg/landmarks_3d_neg_angles.csv'


# saved_path = '/home/baebro/nipa_ws/nipaproj_ws/output/'
saved_path = '/home/baebro/nipa_ws/nipaproj_ws/output/pt files/'
# model_name = 'model_state_dict_test.pt'
# model_name = 'lstm_2000_cat_ov5.pth'
# head_name = 'head_2000_cat_ov5.pth'
# model_name = 'model_state_dict_lstm_2000_add_ov5.pt'
model_name = 'model_state_dict_lstm_mp_full_ep2000.pt'
head_name = 'model_state_dict_head_mp_full_ep2000.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# origin_input = read_angle(origin_path).to(device)
# pos_input = read_angle(pos_path).to(device)
# neg_input = read_angle(neg_path).to(device)

origin_input = read_angles(origin_path, 30).to(device)
pos_input = read_angles(pos_path, 30).to(device)
neg_input = read_angles(neg_path, 30).to(device)


model = skeletonLSTM(12, 64)
head = head().to(device)
model.load_state_dict(torch.load(saved_path + model_name))
head.load_state_dict(torch.load(saved_path + head_name))
model.to(device)
head.to(device)

# model = torch.load(saved_path+model_name).to(device)
# head = torch.load(saved_path+head_name).to(device)
model.eval()
head.eval()

origin_emb = model(origin_input)
pos_emb = model(pos_input)
neg_emb =model(neg_input)

origin_emb_norm = F.normalize(origin_emb, dim=1)
pos_emb_norm = F.normalize(pos_emb, dim=1)
neg_emb_norm = F.normalize(neg_emb, dim=1)

tensor_numpy = origin_emb_norm.detach().cpu().numpy()
np.save(saved_path + 'embedding_vector_mp_full.npy', tensor_numpy)

pos_dist = 1-torch.pow(F.pairwise_distance(origin_emb_norm, pos_emb_norm), 1)/2
neg_dist = 1-torch.pow(F.pairwise_distance(origin_emb_norm, neg_emb_norm), 1)/2


print(neg_dist)
print(pos_dist)
print(neg_dist/pos_dist)

pos_classification = head(torch.add(origin_emb, pos_emb))
neg_classification = head(torch.add(origin_emb, neg_emb))

# pos_classification = head(torch.cat((origin_emb, pos_emb), dim=1))
# neg_classification = head(torch.cat((origin_emb, neg_emb), dim=1))

print(neg_classification)
print(pos_classification)