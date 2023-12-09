from detecto import core, utils, visualize
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 이미지 파일 경로
image_path = "C:\\Users\\user\\Desktop\\swnew\\common\\Suwon_A\\L_2212_Suwon_A_E_C1620\\sensor_raw_data\\camera\\front\\L_2212_Suwon_A_E_C1620_0001.jpg"

# Detecto 모델 로드 (미리 훈련되었다고 가정)
model = core.Model()

# 이미지 불러오기
image = utils.read_image(image_path)

# 객체 감지 수행
labels, boxes, scores = model.predict(image)


visualize.show_labeled_image(image, boxes, labels)


### 일단 이 파일에는 모든 common(원천데이터)를 pretrain 모델에 돌려서 결과를 return 하자

## 이후 detectoRS모델을 학습하는 것이 아닌 학습할수도 없음 그럴빠엔 차라리 다른 이미지 캡션 모델 몇개와 비교하는 식으로 하자 
## ex) yolo ....