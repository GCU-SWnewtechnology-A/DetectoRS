from detecto import core, utils, visualize
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import os
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

# 입력 폴더와 출력 폴더의 상위 경로 설정
input_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\common\\Suwon_A'
output_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\return'
start_folder = 1620
end_folder = 1687



def detectors(input_folder_base, output_folder_base, start_folder, end_folder):
    model = core.Model()
    for folder_name in range(start_folder, end_folder + 1):
        input_folder = os.path.join(input_folder_base, f'L_2212_Suwon_A_E_C{folder_name}', 'sensor_raw_data', 'camera', 'front')
        output_folder = os.path.join(output_folder_base, f'L_2212_Suwon_A_E_C{folder_name}')
        os.makedirs(output_folder, exist_ok=True)
        result_image_folder = os.path.join(output_folder, 'result_images')
        os.makedirs(result_image_folder, exist_ok=True)

        for file_name in os.listdir(input_folder):

            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):

                image_path = os.path.join(input_folder, file_name)
                image = utils.read_image(image_path)
                labels, boxes, scores = model.predict(image)
                
                for label, box in zip(labels, boxes):
                    x, y, x_max, y_max = box
                    cv2.rectangle(image, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.putText(image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                result_image_path = os.path.join(result_image_folder, file_name)
                cv2.imwrite(result_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))




def faster_cnn(input_folder_base, output_folder_base, start_folder, end_folder):
    for folder_name in range(start_folder, end_folder + 1):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        input_folder = os.path.join(input_folder_base, f'L_2212_Suwon_A_E_C{folder_name}', 'sensor_raw_data', 'camera', 'front')
        output_folder = os.path.join(output_folder_base, f'L_2212_Suwon_A_E_C{folder_name}')
        os.makedirs(output_folder, exist_ok=True)

        result_image_folder = os.path.join(output_folder, 'result_images')
        os.makedirs(result_image_folder, exist_ok=True)
        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_folder, file_name)

                img = Image.open(image_path).convert("RGB")
                img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    prediction = model(img_tensor)

                boxes = prediction[0]['boxes'].cpu().numpy()
                labels = prediction[0]['labels'].cpu().numpy()
                
                img_np = cv2.imread(image_path)
                for box, label in zip(boxes, labels):
                    x, y, x_max, y_max = map(int, box)
                    cv2.rectangle(img_np, (x, y), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img_np, f'Label: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                result_image_path = os.path.join(result_image_folder, file_name)
                cv2.imwrite(result_image_path, img_np)


def yolo(input_folder_base, output_folder_base, start_folder, end_folder):
    model = YOLO("yolov8n.pt")
    model.eval()
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

