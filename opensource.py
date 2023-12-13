#%%
from PIL import Image, ImageDraw, ImageFont
import IPython.display as display
from detecto import core, utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torch
import cv2
import os
from ultralytics import YOLO
import warnings
import numpy as np
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def concatenate_images_horizontally(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return new_image

def visualize_results(detector_image, faster_cnn_image, yolov_image):
    detector_image = Image.fromarray(detector_image).resize((640, 480))
    faster_cnn_image = Image.fromarray(faster_cnn_image).resize((640, 480))
    yolov_image = Image.fromarray(yolov_image).resize((640, 480))

    concatenated_image = concatenate_images_horizontally([detector_image, faster_cnn_image, yolov_image])

    draw = ImageDraw.Draw(concatenated_image)
    
    font_size = 25
    font = ImageFont.truetype("arial.ttf", font_size)

    draw.text((10, 10), "DetectoRS", (255, 255, 255), font=font)
    draw.text((detector_image.width + 10, 10), "Faster CNN", (255, 255, 255), font=font)
    draw.text((detector_image.width + faster_cnn_image.width + 10, 10), "YOLOv5", (255, 255, 255), font=font)
    display.display(concatenated_image)


def process_images(input_folder_base, output_folder_base, start_folder, end_folder, models):
    for folder_name in range(start_folder, end_folder + 1):
        input_folder = os.path.join(input_folder_base, f'L_2212_Suwon_A_E_C{folder_name}', 'sensor_raw_data', 'camera', 'front')
        output_folder = os.path.join(output_folder_base, f'L_2212_Suwon_A_E_C{folder_name}')
        os.makedirs(output_folder, exist_ok=True)

        result_image_folder = os.path.join(output_folder, 'result_images')
        os.makedirs(result_image_folder, exist_ok=True)

        result_folder = os.path.join(output_folder, 'result')
        os.makedirs(result_folder, exist_ok=True)

        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_folder, file_name)
                image = utils.read_image(image_path)

                detector_image = None
                faster_cnn_image = None
                yolov_image = None

                if 'detectors' in models:
                    print("---- detectoRS using ----")
                    detector_model = core.Model()
                    labels, boxes, scores = detector_model.predict(image)
                    if image.shape[-1] == 3:  
                        detector_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        detector_image = image

                    for label, box in zip(labels, boxes):
                        x, y, x_max, y_max = box
                        cv2.rectangle(detector_image, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                        cv2.putText(detector_image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    result_image_path = os.path.join(result_image_folder, f'detector_{file_name}')
                    # cv2.imwrite(result_image_path, cv2.cvtColor(detector_image, cv2.COLOR_RGB2BGR))

                if 'faster_cnn' in models:
                    print("---- faster_cnn using ----")
                    faster_cnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
                    faster_cnn_model.eval()
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    faster_cnn_model.to(device)

                    img = Image.open(image_path).convert("RGB")
                    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        prediction = faster_cnn_model(img_tensor)

                    boxes = prediction[0]['boxes'].cpu().numpy()
                    labels = prediction[0]['labels'].cpu().numpy()

                    img_np = cv2.imread(image_path)
                    for box, label in zip(boxes, labels):
                        x, y, x_max, y_max = map(int, box)
                        cv2.rectangle(img_np, (x, y), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(img_np, f'라벨: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    result_image_path = os.path.join(result_image_folder, f'faster_cnn_{file_name}')
                    # cv2.imwrite(result_image_path, img_np)
                    faster_cnn_image = img_np

                if 'yolov' in models:
                    print("---- yolo using ----")
                    yolov_model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
                    yolov_model.eval()

                    img = Image.open(image_path)
                    results = yolov_model(img)
                    result_file_path = os.path.join(result_folder, f"yolov_{os.path.splitext(file_name)[0]}_result.txt")
                    yolov_image = results.render()[0]
                    # yolov_image.save(result_file_path.replace('_result.txt', '_result.jpg'))

                visualize_results(detector_image, faster_cnn_image, yolov_image)

if __name__ == "__main__":
    
    input_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\common\\Suwon_A'   #image Data input folder path
    output_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\return'           #Folder path to return to after image detection in the model 
    
    start_folder = 1620                                                       #Image range settings
    end_folder = 1687
    
    models = ['detectors', 'faster_cnn', 'yolov']                             #Set model name



    process_images(input_folder_base, output_folder_base, start_folder, end_folder, models)

# %%
