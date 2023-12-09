from detecto import core, utils, visualize
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 이미지 파일 경로
image_path = "C:\\Users\\user\\Desktop\\swnew\\common\\2212\\Suwon_A\\E\\L_2212_Suwon_A_E_C1620\\sensor_raw_data\\camera\\front\\L_2212_Suwon_A_E_C1620_0001.jpg"

# Detecto 모델 로드 (미리 훈련되었다고 가정)
model = core.Model()

# 이미지 불러오기
image = utils.read_image(image_path)

# 객체 감지 수행
labels, boxes, scores = model.predict(image)


visualize.show_labeled_image(image, boxes, labels)


### 일단 이 파일에는 모든 common(원천데이터)를 pretrain 모델에 돌려서 결과를 return 하자

## 이후 detectoRS모델을 학습할때 1620 ~ 1687까지 폴더가 있으니 1680까지만 train시키고 나머진 test 후 위에거랑 비교해서 ppt에 넣을 수 있도록 만들면 댈듯?