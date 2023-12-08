import os
import json
from detecto.core import Dataset

# 원본 데이터 폴더 경로
data_folder = r'C:\Users\DR231017\Desktop\deteto_data\01.원천데이터\2212'
json_folder = r'C:\Users\DR231017\Desktop\deteto_data\02.라벨링데이터\2212'

# Detecto 데이터셋 생성
dataset = Dataset()

# 모든 하위 폴더에 대해 반복
for root, dirs, files in os.walk(data_folder):
    for dir_name in dirs:
        # 이미지 파일이 있는 폴더 경로
        image_folder = os.path.join(root, dir_name, 'L_2212_Suwon_A_E_C1651', 'sensor_raw_data', 'camera', 'front')

        # 이미지 파일 탐색
        for file_name in os.listdir(image_folder):
            # 파일 확장자가 이미지인지 확인 (이 부분은 확장자에 따라 수정해야 할 수 있음)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 이미지 파일의 경로
                image_path = os.path.join(image_folder, file_name)

                # 클래스 레이블 및 위치 정보 찾기
                json_path = os.path.join(json_folder, dir_name + '.json')
                with open(json_path, 'r') as json_file:
                    json_data = json.load(json_file)

                    for item in json_data:
                        # 해당 이미지 파일에 대한 정보 찾기
                        if item.get('obj_id') == file_name.split('.')[0]:
                            # 위치 정보
                            position_x = item['psr']['position']['x']
                            position_y = item['psr']['position']['y']

                            # 클래스 레이블
                            label = item['obj_type']

                            # 데이터셋에 추가
                            dataset.add_image(image_path, (position_x, position_y), label=label)

# Detecto 모델 학습
from detecto.core import Model

# 클래스 레이블 설정 (고려된 클래스 레이블 사용)
class_labels = set(item['obj_type'] for item in json_data)

model = Model(class_labels)
model.fit(dataset, epochs=10, learning_rate=0.001, verbose=True)

# 훈련된 모델 저장
model.save('your_trained_model.pth')
