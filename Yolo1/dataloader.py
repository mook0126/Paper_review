import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        """
        Args:
            img_dir (str): 이미지 파일들이 저장된 디렉토리 경로
            label_dir (str): Pascal VOC 라벨(.xml) 파일이 저장된 디렉토리 경로
            S (int): 그리드 개수
            B (int): 박스 예측 개수
            C (int): 클래스 개수 (Pascal VOC는 20개)
            transform (callable, optional): 데이터 변환 함수
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        label_path = os.path.join(
            self.label_dir, self.img_files[index].replace(".jpg", ".xml")
        )

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")

        # 라벨 로드 및 변환
        boxes = self._parse_annotation(label_path)

        if self.transform:
            image = self.transform(image)

        label_matrix = self._convert_to_grid(boxes)

        return image, label_matrix

    def _parse_annotation(self, label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        boxes = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_idx = self._class_to_idx(class_name)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax, class_idx])
        
        return boxes

    def _class_to_idx(self, class_name):
        classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]
        return classes.index(class_name)

    def _convert_to_grid(self, boxes):
        """
        박스를 YOLO 그리드에 맞게 변환
        """
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            xmin, ymin, xmax, ymax, class_idx = box
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            i, j = int(self.S * y_center / 448), int(self.S * x_center / 448)
            x_cell, y_cell = self.S * x_center / 448 - j, self.S * y_center / 448 - i
            width_cell, height_cell = width / 448, height / 448

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, class_idx] = 1

        return label_matrix