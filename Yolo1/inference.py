import torch
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Yolov1
from utils import (
    cellboxes_to_boxes,
    non_max_suppression,
    plot_image,
    load_checkpoint
)

# 경로 설정
WEIGHT_FILE = "D:/YOLO/Yolo1/overfit.pth.tar"  # 저장된 모델 가중치

IMG_SIZE = 448  # YOLO 모델의 입력 이미지 크기
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
def load_model(weight_file, device):
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    checkpoint = torch.load(weight_file, map_location=device)
    load_checkpoint(checkpoint, model)  # Optimizer는 로드할 필요 없음
    model.eval()  # 평가 모드로 전환
    print(f"Model loaded from {weight_file}")
    return model

# 이미지 전처리
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")  # 이미지를 RGB로 변환
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 추론
def inference(model, image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor.to(DEVICE))  # 모델 추론
        predictions = predictions.reshape(-1, 7, 7, 30)  # YOLOv1의 출력 형태로 변환
        bboxes = cellboxes_to_boxes(predictions)  # Bounding Box 변환
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4)
    return bboxes

# 메인 함수
def main(image_path):
    # 모델 로드
    model = load_model(WEIGHT_FILE, DEVICE)

    # 이미지 전처리
    image_tensor = preprocess_image(image_path)

    # 추론
    bboxes = inference(model, image_tensor)

    # 결과 시각화
    image = Image.open(image_path).convert("RGB")
    plot_image(image, bboxes)  # Bounding Box와 클래스 시각화
    plt.show()

if __name__ == "__main__":
    test_image_path = "D:/YOLO/Yolo1/test_image.jpg"  # 테스트할 이미지 경로
    main(test_image_path)