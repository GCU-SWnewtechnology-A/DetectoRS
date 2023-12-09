from detecto import core, utils, visualize
import warnings
import cv2
import os
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 입력 폴더와 출력 폴더의 상위 경로 설정
input_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\common\\Suwon_A'
output_folder_base = r'C:\\Users\\user\\Desktop\\swnew\\return'

model = core.Model()

# 폴더 범위 설정
start_folder = 1620
end_folder = 1687

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
