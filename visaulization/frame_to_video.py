import cv2
import os
import re

def extract_number(filename):
    # 파일 이름에서 숫자 부분만 추출 (예: '042_E00_001_L_0000660'에서 660 추출)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # 숫자가 없으면 무한대 반환하여 뒤로 정렬

def images_to_video(image_folder, output_video_path, fps=30):
    # 이미지 파일 목록을 숫자 순서로 정렬
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
        key=extract_number  # 숫자만 추출하여 정렬
    )

    if not images:
        print("Error: No images found in the specified folder.")
        return

    # 첫 번째 이미지로부터 영상의 크기 정보를 얻음
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape

    # 비디오 코덱 및 출력 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID' 사용 가능
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 모든 이미지를 프레임으로 추가
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read {image}, skipping.")
            continue
        video.write(frame)

    # 리소스 해제
    video.release()
    print(f"Video saved successfully at {output_video_path}")

# 사용 예시
# image_folder = "/path/to/image/folder"
# output_video_path = "/path/to/output/video.mp4"
# images_to_video(image_folder, output_video_path, fps=30)




# 사용 예시
# image_folder = "/home/baebro/nipa_ws/nipaproj_ws/visaulization/sample_data/camera4/visualization"
image_folder = "/media/baebro/NIPA_data/Train/[원천]빨간맛1 (레드벨벳).zip_unzipped/042_E00_001_P"
output_video_path = "/home/baebro/nipa_ws/nipaproj_ws/visaulization/sample_data_video/video_rvv_P.mp4"
images_to_video(image_folder, output_video_path, fps=30)
