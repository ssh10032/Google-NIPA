# Google-NIPA 프로젝트
---
## 모델 아키텍쳐
![image](https://github.com/user-attachments/assets/bf722219-088e-4000-82e3-10a6f46a9b2d)
  +
  +

---
## 학습 데이터
![image](https://github.com/user-attachments/assets/ac9ae31c-6f37-4f53-a1f3-7036d2fc76be)
  + AI hub의 총 100 x 3 개 k pop 영상 사용 (안무 x 각도)
  + 숏폼은 같은 전방 영상이라도 카메라 각도가 다름을 확인

---
## 데이터 전처리
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
## 손실 함수
![image](https://github.com/user-attachments/assets/02630cf7-e806-4abd-b427-4d90fd7d9980)
  + Supervised Contrastive Learning(SCL) Loss + BCE Loss

---
## 서비스 아키텍처
![image](https://github.com/user-attachments/assets/319ec31d-5655-46bf-8faa-e191226911c6)

---
## 프로젝트 개발 프로세스
![image](https://github.com/user-attachments/assets/5d2a5536-32ff-4ade-8bc6-531bc46e5ccc)

---
## 시연 영상
![image](https://github.com/user-attachments/assets/017dfd9c-db48-4abf-ac7a-64adb908b98f)

---
## 개발 환경
![image](https://github.com/user-attachments/assets/ab2a74a7-0cee-47b1-98e5-24b1e00bff28)
  + <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> 
  + OS : Ubuntu 20.04.6 LTS 
  + GPU : NVIDIA GeForce RTX 3080 Ti x 2
  + CPU : i9-10940X
  + CUDA : 12.6
    

