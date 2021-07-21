# engocr

yolov3+crnn ocr

현재(210721) yolov3모델 훈련 덜됨(20에포크에서 스탑,너무더워서 못돌리겠음)
            crnn모델 valimage를 train으로,val을 train에서 일부 떼서 사용(메모리가 부족해서 train 못돌렸음)
순서:
  1. 데이터세트 다운로드(전 textocr사용했음)
  2. yolo/yolo3_json_preprocessing.ipynb 로 yolo용 데이터생성(train,val 다해줘야됨)
  3. yolo/yolov3/train_Mobilenet 로 yolov3모델생성(모바일넷 학습잠그고 30에포크 풀고 20에포크돌리는듯)
  4. crnn_preprocessing.ipynb 로 crnn cropjpg 과 label json생성 (train val 다해줘야됨)
  5. 밑의 ssd디텍터에서 crnn가중치파일 다운로드후 crnnmodeltest.ipynb에서 훈련,v5까지 잠그고 오버핏날때까지 훈련후 풀고 오버핏날때까지 훈련
  6. yolocrnn.ipynb에서 테스트


기타:
  한국어파일을 다운받긴했는데



참고:
  https://github.com/Adamdad/keras-YOLOv3-mobilenet

  https://textvqa.org/textocr/dataset

  https://github.com/mvoelk/ssd_detectors

  https://wiserloner.tistory.com/
