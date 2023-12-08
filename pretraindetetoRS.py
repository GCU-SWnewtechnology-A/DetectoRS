from detecto import core, utils, visualize

# 이미지 파일 경로
image_path = "C:\\Users\\DR231017\\Desktop\\deteto_data\\01.원천데이터\\2212\\L_2212_Suwon_A_E_C1622\\sensor_raw_data\\camera\\front\\L_2212_Suwon_A_E_C1622_0003.jpg"

# Detecto 모델 로드 (미리 훈련되었다고 가정)
model = core.Model()

# 이미지 불러오기
image = utils.read_image(image_path)

# 객체 감지 수행
labels, boxes, scores = model.predict(image)

# 결과 시각화
visualize.show_labeled_image(image, boxes, labels)


import cv2
from matplotlib import pyplot as plt
import os

image_path = "C:\\Users\\DR231017\\Desktop\\deteto_data\\01.원천데이터\\2212\\L_2212_Suwon_A_E_C1622\\sensor_raw_data\\camera\\front\\L_2212_Suwon_A_E_C1622_0003.jpg"

# 이미지 파일이 존재하는지 확인
if not os.path.isfile(image_path):
    print("hello")
    print("Error: Image file does not exist.")
else:
    # 이미지 파일 불러오기
    image = cv2.imread(image_path)

    # 이미지가 정상적으로 불러와졌는지 확인
    if image is None:
        print("Error: Failed to load the image.")
    else:
        # 이미지 출력
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
