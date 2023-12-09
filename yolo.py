import torch
from pathlib import Path
from PIL import Image
import os
from ultralytics import YOLO

#model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
model = YOLO("yolov8n.pt")
model.eval()

# 입력 폴더와 출력 폴더의 상위 경로 설정
input_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\common\\Suwon_A'
output_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\return'

# 폴더 범위 설정
start_folder = 1620
end_folder = 1660
for folder_name in range(start_folder, end_folder + 1):
    input_folder = os.path.join(input_folder_base, f'L_2212_Suwon_A_E_C{folder_name}', 'sensor_raw_data', 'camera', 'front')
    output_folder = os.path.join(output_folder_base, f'L_2212_Suwon_A_E_C{folder_name}')
    os.makedirs(output_folder, exist_ok=True)

    result_folder = os.path.join(output_folder, 'result')
    os.makedirs(result_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, file_name)
            img = Image.open(image_path)
            results = model(img)
            result_file_path = os.path.join(result_folder, f"{os.path.splitext(file_name)[0]}_result.txt")

            results.save(save_dir=result_folder)

