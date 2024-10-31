import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from dataloader.dataloader import LandmarkDataset, ContrastiveDataset
from model import skeleton_LSTM, skeletonLSTM, skeletonLSTM_bc
from loss import TripletContrastiveLoss, ContrastiveLoss
from tqdm import tqdm

def main():
    wandb.init(project="skeleton_lstm_new", name="experiment1")
    save_path = '/home/baebro/nipa_ws/nipaproj_ws/output/'
    # 하이퍼파라미터 설정
    feature_dim = 12
    output_dim = 64
    num_epochs = 2000  # Early stopping 적용 시 더 큰 값을 설정해도 됩니다
    learning_rate = 0.0001
    temperature = 0.9
    patience = 25  # Early stopping patience 설정
    min_delta = 0.001  # Validation loss가 감소하는 최소 값
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 데이터셋과 데이터로더
    dataset = ContrastiveDataset(15)
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
        shuffle=False,  # 데이터 순서를 섞어서 학습 효과를 높임
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
    model = skeletonLSTM_bc(feature_dim, output_dim).to(device)

    criterion = ContrastiveLoss(margin=1.0)
    # criterion = TripletContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Early stopping 변수 초기화
    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    # 학습 루프
    for epoch in range(num_epochs):
        if early_stop:
            break

        model.train()
        epoch_loss = 0.0
        train_loader_tqdm = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")
        for batch_idx, (anchor, pos, neg) in enumerate(train_loader_tqdm):
            optimizer.zero_grad()
            # 데이터를 장치로 이동
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            # 모델에 통과하여 임베딩 생성
            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)

            # Contrastive Loss 계산
            positive_dist = F.pairwise_distance(anchor_emb, pos_emb)
            negative_dist = F.pairwise_distance(anchor_emb, neg_emb)
            loss = criterion(positive_dist, negative_dist)

            # TripletContrastiveLoss 계산
            # loss = criterion(anchor_emb, pos_emb, neg_emb)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loader_tqdm.set_postfix(loss=loss.item())

        scheduler.step()  # 학습률 조정

        # 에폭마다 평균 손실을 기록
        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Validation")
        with torch.no_grad():
            for batch_idx, (anchor, pos, neg) in enumerate(val_loader_tqdm):
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

                # 임베딩 생성
                anchor_emb = model(anchor)
                pos_emb = model(pos)
                neg_emb = model(neg)

                # Contrastive Loss 계산
                positive_dist = F.pairwise_distance(anchor_emb, pos_emb)
                negative_dist = F.pairwise_distance(anchor_emb, neg_emb)
                loss = criterion(positive_dist, negative_dist)

                # Validation loss 계산 - TripletContrastiveLoss
                # loss = criterion(anchor_emb, pos_emb, neg_emb)

                val_loss += loss.item()
                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

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
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

    print("Training complete!")
    torch.save(model.state_dict(), save_path +'model_state_dict_Contrastive_loss_2000.pt')

if __name__ == '__main__':
    main()
