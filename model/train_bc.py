import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from dataloader.dataloader import LandmarkDataset, ContrastiveDataset
from model import skeleton_LSTM, skeletonLSTM, head
from loss import TripletContrastiveLoss, ContrastiveLoss
from tqdm import tqdm
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(12)

def main():
    wandb.init(project="skeleton_lstm_new", name="experiment1")
    save_path = '/home/baebro/nipa_ws/nipaproj_ws/output/'
    # 하이퍼파라미터 설정
    feature_dim = 12
    output_dim = 64
    num_epochs = 10  # Early stopping 적용 시 더 큰 값을 설정해도 됩니다
    learning_rate = 0.0001
    temperature = 0.1
    patience = 25  # Early stopping patience 설정
    min_delta = 0.001  # Validation loss가 감소하는 최소 값
    sequence_length = 30

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 데이터셋과 데이터로더
    dataset = ContrastiveDataset(sequence_length)
    # print(len(dataset))
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    # Subset을 사용하여 학습 및 검증 데이터셋 생성
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # DataLoader 생성
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=0,  # CPU 데드락 방지를 위해 num_workers=0으로 설정
        batch_size=8,
        shuffle=True,  # 데이터 순서를 섞어서 학습 효과를 높임
        pin_memory=True  # GPU 사용 시 유용
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=0,  # CPU 데드락 방지를 위해 num_workers=0으로 설정
        batch_size=8,  # 검증도 충분한 배치 크기로 설정
        shuffle=False,
        pin_memory=True  # GPU 사용 시 유용
    )
    print()

    # 모델, 손실 함수, 옵티마이저, 스케줄러 초기화
    # model = skeletonLSTM(feature_dim, output_dim).to(device)
    model = skeletonLSTM(feature_dim, output_dim).to(device)
    # binary classification head
    classification = head().to(device)

    # criterion1 = ContrastiveLoss(margin=1.0)
    criterion1 = TripletContrastiveLoss(temperature=temperature)

    # Binary classification loss
    criterion2 = nn.BCELoss()
    criterion3 = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Early stopping 변수 초기화
    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False
    pos1 = pd.read_csv(
        '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/labeled_data/pos1_infer/landmarks_3d_pos1_infer_angles.csv')
    pos2 = pd.read_csv(
        '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/labeled_data/pos2_infer/landmarks_3d_pos2_infer_angles.csv')
    neg = pd.read_csv(
        '/home/baebro/nipa_ws/nipaproj_ws/sample_videos/labeled_data/neg_infer/landmarks_3d_neg_infer_angles.csv')
    pos1_list = []
    pos2_list = []
    neg_list = []
    for i in range(0, 300, sequence_length):
        pos1_list.append(torch.tensor(pos1.iloc[i:i + sequence_length].values, dtype=torch.float32))
        pos2_list.append(torch.tensor(pos2.iloc[i:i + sequence_length].values, dtype=torch.float32))
        neg_list.append(torch.tensor(neg.iloc[i:i + sequence_length].values, dtype=torch.float32))

    pos1_tensor = torch.stack(pos1_list, dim=0).to(device)
    pos2_tensor = torch.stack(pos2_list, dim=0).to(device)
    neg_tensor = torch.stack(neg_list, dim=0).to(device)

    # 학습 루프
    for epoch in range(num_epochs):
        if early_stop:
            break

        model.train()
        classification.train()

        epoch_loss = 0.0
        p_epoch_loss = 0.0
        n_epoch_loss = 0.0
        # epoch_classification_loss = 0.0
        train_loader_tqdm = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")
        for batch_idx, (anchor, pos, neg) in enumerate(train_loader_tqdm):

            optimizer.zero_grad()
            # 데이터를 장치로 이동
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            if batch_idx % 3==0:
                anchor = torch.cat((anchor,pos1_tensor),dim=0)
                pos = torch.cat((pos, pos2_tensor), dim=0)
                neg = torch.cat((neg, neg_tensor), dim=0)

            # 모델에 통과하여 임베딩 생성 및 이진 분류 출력
            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)

            # Contrastive Loss 계산
            # anchor_emb = F.normalize(anchor_emb, dim=1)
            # pos_emb = F.normalize(pos_emb, dim=1)
            # neg_emb = F.normalize(neg_emb, dim=1)

            # positive_dist = F.pairwise_distance(anchor_emb, pos_emb)
            # negative_dist = F.pairwise_distance(anchor_emb, neg_emb)
            # loss1 = criterion1(positive_dist, negative_dist)

            loss1 = criterion1(anchor_emb, pos_emb, neg_emb)

            # Classification Loss 계산
            pos_classification = classification(torch.add(anchor_emb, pos_emb))
            loss2 = criterion2(pos_classification, torch.full((pos_classification.shape[0], 1), 1.).to(device))

            neg_classification = classification(torch.add(anchor_emb, neg_emb))
            loss3 = criterion3(neg_classification, torch.full((neg_classification.shape[0], 1), 0.).to(device))

            # Total loss 계산
            loss = loss1 + 1.5 * (loss2 + loss3)

            epoch_loss += loss.item()
            p_epoch_loss += loss2.item()
            n_epoch_loss += loss3.item()
            # epoch_classification_loss += classification_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loader_tqdm.set_postfix(loss=loss.item(), contrastive_loss=loss1.item(), P_classification_loss=loss2.item(), N_classification_loss=loss3.item())


        scheduler.step()  # 학습률 조정

        # 에폭마다 평균 손실을 기록
        avg_train_contrastive_loss = loss1.item() / len(train_dataloader)
        avg_train_Pos_bc_loss = p_epoch_loss / len(train_dataloader)
        avg_train_Neg_bc_loss = n_epoch_loss / len(train_dataloader)

        avg_train_loss = epoch_loss / len(train_dataloader)
        # avg_classification_loss = epoch_classification_loss / len(train_dataloader)
        # print(
        #     f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, P_Classification Loss: {avg_train_Pos_bc_loss:.4f}, N_Classification Loss: {avg_train_Neg_bc_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_p_loss = 0.0
        val_n_loss = 0.0
        # val_classification_loss = 0.0
        val_loader_tqdm = tqdm(val_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Validation")
        with torch.no_grad():
            for batch_idx, (anchor, pos, neg) in enumerate(val_loader_tqdm):
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

                # 임베딩 생성 및 이진 분류 출력
                anchor_emb = model(anchor)
                pos_emb = model(pos)
                neg_emb = model(neg)

                # Contrastive Loss 계산
                # anchor_emb = F.normalize(anchor_emb, dim=1)
                # pos_emb = F.normalize(pos_emb, dim=1)
                # neg_emb = F.normalize(neg_emb, dim=1)
                # positive_dist = F.pairwise_distance(anchor_emb, pos_emb)
                # negative_dist = F.pairwise_distance(anchor_emb, neg_emb)
                # loss1 = criterion1(positive_dist, negative_dist)

                loss1 = criterion1(anchor_emb, pos_emb, neg_emb)

                # Classification Loss 계산
                pos_classification = classification(torch.add(anchor_emb, pos_emb))
                loss2 = criterion2(pos_classification, torch.full((pos_classification.shape[0], 1), 1.).to(device))

                neg_classification = classification(torch.add(anchor_emb, neg_emb))
                loss3 = criterion3(neg_classification, torch.full((neg_classification.shape[0], 1), 0.).to(device))

                # Total validation loss 계산
                loss = loss1 + 1.5 * (loss2 + loss3)

                val_loss += loss.item()
                val_p_loss += loss2.item()
                val_n_loss += loss3.item()

                val_loader_tqdm.set_postfix(loss=loss.item(), contrastive_loss=loss1.item(), P_classification_loss=loss2.item(), N_classification_loss=loss3.item())

        # avg_train_contrastive_loss = loss1.item() / len(train_dataloader)
        # avg_train_Pos_bc_loss = loss2.item() / len(train_dataloader)
        # avg_train_Neg_bc_loss = loss3.item() / len(train_dataloader)

        # print(
        #     f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_contrastive_loss:.4f}, P_Classification Loss: {avg_train_Pos_bc_loss:.4f}, N_Classification Loss: {avg_train_Neg_bc_loss:.4f}')

        avg_val_loss = val_loss / len(val_dataloader)
        # avg_val_classification_loss = val_classification_loss / len(val_dataloader)

        # avg_val_loss = val_loss.item() / len(train_dataloader)
        avg_val_Pos_bc_loss = val_p_loss / len(train_dataloader)
        avg_val_Neg_bc_loss = val_n_loss / len(train_dataloader)

        # print(
        #     f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, P_Classification Loss: {avg_val_Pos_bc_loss:.4f}, , N_Classification Loss: {avg_val_Neg_bc_loss:.4f}')

        # Early Stopping 체크
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Improvement이 있으면 카운트 리셋
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            early_stop = True

        # 에폭마다 평균 손실을 기록
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss,
                   "val_loss": avg_val_loss})

    print("Training complete!")
    # torch.save(model.state_dict(), save_path + 'model_state_dict_Contrastive_loss_w_bc.pt')
    # torch.save(classification.state_dict(), save_path + 'model_state_dict_Contrastive_loss_w_bc.pt')
    torch.save(model, save_path + 'lstm_2000_ov5.pth')
    torch.save(classification, save_path + 'head_2000_ov5.pth')



if __name__ == '__main__':
    main()