from pathlib import Path
import torch
from PIL import Image
import pandas as pd


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        # 이전 코드: os.path 사용
        # self.img_dir = img_dir
        # self.label_dir = label_dir
        
        # 변경: pathlib 사용
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 이전 코드: os.path.join
        # label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        
        # 변경: pathlib 사용
        label_path = self.label_dir / self.annotations.iloc[index, 1]
        img_path = self.img_dir / self.annotations.iloc[index, 0]

        # 이전 코드: 파일 존재 여부 확인 없음
        # 변경: 파일 존재 여부 확인 추가
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Read labels
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                # 이전 코드: 데이터 읽기만 수행
                # 변경: 숫자 변환 로직 및 float->int 변환 추가
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.strip().split()
                ]
                boxes.append([class_label, x, y, width, height])

        # 변경: 빈 boxes 처리
        if len(boxes) == 0:
            raise ValueError(f"No bounding boxes found in {label_path}")

        # 이전 코드: 단순 이미지 읽기
        # image = Image.open(img_path)
        
        # 변경: 예외 처리 및 RGB 변환
        try:
            image = Image.open(img_path).convert("RGB")  # RGB로 변환
        except Exception as e:
            raise RuntimeError(f"Failed to open image file {img_path}: {e}")

        boxes = torch.tensor(boxes)

        # 이전 코드: transform 호출 주석 처리
        # if self.transform:
        #     image = self.transform(image)

        # 변경: transform 실행 시 예외 처리 추가
        if self.transform:
            try:
                image, boxes = self.transform(image, boxes)
            except Exception as e:
                raise RuntimeError(f"Transform failed on {img_path}: {e}")

        # Initialize label matrix
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # 이전 코드: width와 height 계산 로직은 동일
            width_cell, height_cell = width * self.S, height * self.S

            # 이전 코드: 중복 확인 없이 바로 데이터 입력
            # if label_matrix[i, j, 20] == 0:
            
            # 변경: 중복 체크 주석 추가
            # NOTE: Supports only ONE object per cell
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
