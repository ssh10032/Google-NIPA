# Google-NIPA 프로젝트


---
# 팀원
![팀원](https://github.com/user-attachments/assets/95218c3b-91dd-40ea-b0a3-6a2a6919e6cb)


---
# 모델 아키텍쳐
![image](https://github.com/user-attachments/assets/bf722219-088e-4000-82e3-10a6f46a9b2d)

---
# 학습 데이터
![데이터소개](https://github.com/user-attachments/assets/ccb2819d-d0a8-4b14-9184-6c8fbdd0012b)
  + AI hub의 총 100 x 3 개 k pop 영상 사용 (안무 x 각도)
  + 숏폼은 같은 전방 영상이라도 카메라 각도가 다름을 확인

---
# 데이터 전처리
![image](https://github.com/user-attachments/assets/e4dc087b-844d-4f3d-952e-4a277eddbbfd)


1. Landmark feature
    + 영상 데이터 → Media pipe (라이브러리) 적용 → 프레임 별 Landmark 추출 → 스켈레톤 벡터 계산 → 신체 부위 각도 계산


2. Feature selection
    + 33개의 landmark 중 14개의 좌표 사용 → 결측치 제거 와 이상치 완화, 성능 향상
    + 최종 input: 신체 각도 12개


3. Sequential data
    + 시퀀스 길이 만큼 동영상 프레임을 결합
    + 시퀀스 길이 30 설정 → Source 데이터와 숏 폼 영상들 30 fps → 1초에 한 번씩 추론
---
# 손실 함수
![손실함수](https://github.com/user-attachments/assets/a01e1500-50fd-4a99-9051-ea1f8ad07606)
  + Supervised Contrastive Learning(SCL) Loss + BCE Loss

---
# 서비스 아키텍처
![서비스아키텍쳐](https://github.com/user-attachments/assets/2805007e-9ab1-44f1-90a5-a1d52f9cd9ec)


---
# 프로젝트 개발 프로세스
![개발프로세스](https://github.com/user-attachments/assets/e519a869-233c-4697-8390-93ed55c21d0a)


---
# 시연 영상
+ 삐끼삐끼 챌린지
![image](https://github.com/user-attachments/assets/56530f55-5da5-4365-9c18-526fe5d02b6d)
+ 마라탕후루 챌린지
![image](https://github.com/user-attachments/assets/d8cc09a4-4eb2-494f-8a40-05400982bccf)


---
# 개발 환경
![개발환경](https://github.com/user-attachments/assets/8ad4b474-a18d-4ab7-b536-7141bfa985da)
+ <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> 
+ OS : Ubuntu 20.04.6 LTS 
+ GPU : NVIDIA GeForce RTX 3080 Ti x 2
+ CPU : i9-10940X
+ CUDA : 12.6
