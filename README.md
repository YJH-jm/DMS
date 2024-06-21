# DMS
DMS (Drive Monitoring System)의 일부 기능을 구현한 코드 <br>
- DMS에는 졸음 경보, 전방 미주시, 피로, 흡연, 핸드폰 사용 등을 감지하는 다양한 기능이 존재 

<br>

여러 기능 중 운전자의 얼굴의 인식하고, 인식한 얼굴의 landmark를 찾아 이를 이용하여 전방을 주시하고 있는지, 졸음 운전을 하고 있는지 확인하고 경고를 출력하는 코드 구현<br>

<br>

Detection 모델을 이용하여 운전자의 얼굴을 감지할 수 있도록 하고 이 때, 얼굴이 감지되지 않으면 전방을 주시하지 않았다고 판단함 <br>
이 때, 사용한 모델은 RetinaFace이고  `facedetection` 폴더에서 확인 가능
- [RetinaFace](https://github.com/YJH-jm/Pytorch_Retinaface)에서 필요한 코드만 가지고 옴
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) 데이터를 이용하여 [RetinaFace](https://github.com/YJH-jm/Pytorch_Retinaface)의 `train.py` 을 학습


<br>

찾은 운전자의 얼굴에서 landmark를 추출 <br>
이 때, PFLD 모델을 학습하여 landmark detection을 진행하였으며, `lmdetection` 폴더에서 확인 가능
- [PFLD](https://github.com/polarisZhao/PFLD-pytorch) 에서 필요한 코드만 가지고 옴
- [WFLW Wider Facial Landmarks in-the-wild](https://wywu.github.io/projects/LAB/WFLW.html)의 데이터를 이용하여 [PFLD](https://github.com/polarisZhao/PFLD-pytorch)의 `train.py` 를 학습

<br>

찾은 landmark 좌표들을 가지고 눈을 뜨고 있는지 (EAR), 고개를 옆으로 돌리고 있는지 (SR), 고개를 들고 있는지(HR)를 계산하여, 특정 threshold 값 범위에서 벗어나면 졸음운전을 한다고 판단
- `calc_lm.py` 코드에 구현

<br>
<br>

## 설치
1. 가상 환경 설정 
    ```
    conda create -n dms python=3.8
    conda activate dms
    ```

<br>

2. pytorch 설치
- Cuda version에 맞는 torch, torchvision 설치

<br>

3. library 설치
    ```
    pip install matplotlib 
    ```
<br>
<br>

## 실행
- 카메라 데이터 사용 
    ```
    python cam_demo.py
    ```
<br>

- 테스트 이미지 사용
    ```
    python inference.py <img file/folder path>
    ```
<br>
Threshold 값은 데이터와 카메라 각도에 맞게 조절 
<br>
<br>

## 결과
AI hub의 [졸음운전 예방을 위한 운전자 상태 정보 ](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=173) 데이터를 이용하여 시각화 하여 확인 
<br>
아래 이미지는 위의 데이터의 샘플데이터의 결과
<br>

<p align=center><img src="./images/1.png" width=80%></p>

<br>

마스크와 안경을 쓰고 있어도 얼굴을 잘 찾아내었으나, landmark의 경우는 정확도가 떨어짐 <br>
또한, 카메라의 각도에 따라서 threshold 을 조절하지 않으면 경고 메세지를 정확하게 출력하지 못함하기 때문에 여러 번의 테스트 후 적절한 값을 찾아야 함 <br>

또한, 옆으로 고개를 돌리면 눈의 비율을 제대로 측정할 수 없기 때문에 눈을 감고 있다고 판단할 가능성이 높음 <br>
이를 처리해주는 방법도 추가할 필요성 있음 <br>

<br>
<br>


