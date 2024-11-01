import torch
from model import skeletonLSTM  # 모델 클래스 정의가 포함된 파일에서 불러옵니다.

saved_path = '/home/baebro/nipa_ws/nipaproj_ws/output/pt files/'
model_name = 'model_state_dict_lstm_2000_add_ov5.pt'
head_name = 'model_state_dict_head_2000_add_ov5.pt'
# 하이퍼파라미터 설정 (모델을 재구성할 때 필요)
feature_dim = 12
output_dim = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 모델 인스턴스 생성 및 가중치 로드
model = skeletonLSTM(feature_dim, output_dim).to(device)
model.load_state_dict(torch.load(saved_path + model_name))

# 모델 평가 모드로 설정 (TorchScript 변환 전에 설정 필요)
model.eval()

# TorchScript로 모델 변환 (trace나 script 방식 선택)
example_input = torch.randn(1, 15, feature_dim).to(device)  # 추론 시 사용하는 입력 예제
traced_model = torch.jit.trace(model, example_input)

# TorchScript 모델을 파일로 저장
torch.jit.save(traced_model, saved_path + 'traced_model_script.pt')
