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
CSV_FILE_TRAIN = BASE_DIR / "data/train.csv"
CSV_FILE_TEST = BASE_DIR / "data/test.csv"
LOAD_MODEL_FILE = BASE_DIR / "overfit.pth.tar"

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False  # 가중치를 로드할지 여부
TEST_EVAL_INTERVAL = 10  # Test 검증 주기 (epoch 단위)
LOG_FILE = BASE_DIR / "training_log.txt"  # 로그 파일 경로

# 데이터 변환
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

# Mixed Precision Training을 위한 Scaler
scaler = torch.cuda.amp.GradScaler()


# 훈련 함수
def train_fn(train_loader, model, optimizer, loss_fn, epoch):
    print(f"Using device: {DEVICE}")
    model.train()
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch [{epoch + 1}]")
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Mixed Precision 사용
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)

        mean_loss.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    mean_epoch_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss for Epoch {epoch + 1}: {mean_epoch_loss}")
    return mean_epoch_loss


# 로그 기록 함수
def log_results(epoch, train_loss, train_map=None, test_map=None):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"Epoch: {epoch + 1}\n")
        log_file.write(f"Train Loss: {train_loss:.4f}\n")
        if train_map is not None:
            log_file.write(f"Train mAP: {train_map:.4f}\n")
        if test_map is not None:
            log_file.write(f"Test mAP: {test_map:.4f}\n")
        log_file.write("-" * 30 + "\n")


# 메인 함수
def main():
    # 로그 파일 초기화
    with open(LOG_FILE, "w") as log_file:
        log_file.write("Training Log\n")
        log_file.write("=" * 30 + "\n")

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # 모델 가중치 로드
    if LOAD_MODEL:
        checkpoint = torch.load(LOAD_MODEL_FILE, map_location=DEVICE)
        load_checkpoint(checkpoint, model, optimizer)

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
        shuffle=False,
        drop_last=False,
    )

    best_map = 0

    for epoch in range(EPOCHS):
        print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, epoch)

        # Train mAP 계산
        train_pred_boxes, train_target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        train_map = mean_average_precision(
            train_pred_boxes, train_target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP for Epoch {epoch + 1}: {train_map:.4f}")

        # Test 검증: 10 Epoch마다 수행
        test_map = None
        if (epoch + 1) % TEST_EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            print(f"\nEvaluating Test set at Epoch {epoch + 1}...")
            test_pred_boxes, test_target_boxes = get_bboxes(
                test_loader, model, iou_threshold=0.5, threshold=0.4
            )
            test_map = mean_average_precision(
                test_pred_boxes, test_target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Test mAP for Epoch {epoch + 1}: {test_map:.4f}")

            # Checkpoint 저장 조건
            if test_map > best_map:
                print(f"New best Test mAP: {test_map:.4f}, saving checkpoint...")
                best_map = test_map
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename=f"model_epoch{epoch+1}_map{test_map:.4f}.pth.tar")
            else:
                print(f"Test mAP did not improve. Current Test mAP: {test_map:.4f}, Best Test mAP: {best_map:.4f}")

        # 로그 저장
        log_results(epoch, train_loss, train_map, test_map)


# 메인 함수 실행
if __name__ == "__main__":
    main()
