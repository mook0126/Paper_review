import os
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

# 작업 디렉토리를 고정
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "data/images"
LABEL_DIR = BASE_DIR / "data/labels"
CSV_FILE_TRAIN = BASE_DIR / "data/100examples.csv"
CSV_FILE_TEST = BASE_DIR / "data/test.csv"
LOAD_MODEL_FILE = BASE_DIR / "overfit.pth.tar"

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False  # 가중치를 로드할지 여부

# 데이터 변환
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


# 훈련 함수
def train_fn(train_loader, model, optimizer, loss_fn, epoch):
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch [{epoch + 1}]")  # Epoch 정보 추가
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss for Epoch {epoch + 1}: {sum(mean_loss) / len(mean_loss)}")


# 메인 함수
def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # 모델 가중치 로드
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # 데이터셋 정의
    train_dataset = VOCDataset(
        csv_file=str(CSV_FILE_TRAIN),
        transform=transform,
        img_dir=str(IMG_DIR),
        label_dir=str(LABEL_DIR),
    )

    test_dataset = VOCDataset(
        csv_file=str(CSV_FILE_TEST),
        transform=transform,
        img_dir=str(IMG_DIR),
        label_dir=str(LABEL_DIR),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    best_map = 0  # 최고 mAP를 저장하는 변수

    for epoch in range(EPOCHS):
        print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}")  # 명시적으로 출력
        train_fn(train_loader, model, optimizer, loss_fn, epoch)

        # 평균 평균 정밀도 계산
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP for Epoch {epoch + 1}: {mean_avg_prec:.4f}")

        # 이전 최고 mAP보다 높으면 가중치 저장
        if mean_avg_prec > best_map:
            print(f"New best mAP: {mean_avg_prec:.4f}, saving checkpoint...")
            best_map = mean_avg_prec  # 최고 mAP 갱신
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=str(LOAD_MODEL_FILE))
        else:
            print(f"mAP did not improve. Current mAP: {mean_avg_prec:.4f}, Best mAP: {best_map:.4f}")


# 메인 함수 실행
if __name__ == "__main__":
    main()
