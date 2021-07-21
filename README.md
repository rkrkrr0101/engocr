# engocr

yolov3+crnn ocr

현재상황(210721):

            yolov3모델 훈련 덜됨(20에포크에서 스탑,너무더워서 못돌리겠음)
            crnn모델 valimage를 train으로,val을 train에서 일부 떼서 사용(메모리가 부족해서 train 못돌렸음),5,7에포크에서 오버핏나는거보니까 데이터셋 엄청작아서 바꾸면 성능오르긴할거같음
            
            
            
순서:

      1. 데이터세트 다운로드(전 textocr사용했음  https://textvqa.org/textocr/dataset  참고)
      2. yolo/yolo3_json_preprocessing.ipynb 로 yolo용 데이터생성(train,val 다해줘야됨)
      3. yolo/yolov3/train_Mobilenet 로 yolov3모델생성(모바일넷 학습잠그고 30에포크 풀고 20에포크돌리는듯)
      4. crnn_preprocessing.ipynb 로 crnn cropjpg 과 label json생성 (train val 다해줘야됨)
      5. 밑의 ssd디텍터에서 crnn가중치파일 다운로드후 crnnmodeltest.ipynb에서 훈련,v5까지 잠그고 오버핏날때까지 훈련후 풀고 오버핏날때까지 훈련
      6. yolocrnn.ipynb에서 테스트
      
      google translate eng
      1. Download dataset (I used textocr, see https://textvqa.org/textocr/dataset)
      2. Create data for yolo with yolo/yolo3_json_preprocessing.ipynb (train and val must be completed)
      3. Create a yolov3 model with yolo/yolov3/train_Mobilenet (like locking mobile net training, unlocking the fork at 30, and running the fork at 20)
      4. Create crnn cropjpg and label json with crnn_preprocessing.ipynb (train val must be completed)
      5. After downloading the crnn weight file from the ssd detector below, training in crnnmodeltest.ipynb, locking up to v5, training until overfitting, releasing and      training until overfitting
      6. Test in yolocrnn.ipynb





기타:

     한국어데이터셋을 다운받긴했는데(https://aihub.or.kr/aidata/133/download aihub페이지)
     시작부터 라벨링 틀리고 데이터도 너무커서 다른데이터셋 찾아서 영어만 사용했음
  
     다음엔 한국어데이터셋 좀 보고 라벨링 정확한지 확인해보고 한국어버전 만들어볼까싶음
  
     그리고 이미지비전에서 전이학습 안하면(최소한 ocr은)진짜 지역최적점 못벗어난다는걸 알았음
      (전부 blank값으로 채운거를 벗어날수가없음 30회정도로는,ctcloss기준(K.ctc_batch_cost 사용시) loss가 15~16일때 전부 blank로 채운값)
  


앞으로 개선할만한 사항:
  
            1.모델 훈련 마저하기
  
            2.모델 변경(어텐션이나 셀프어텐션계열,yolov5   bert,트랜스포머)
  
            3.전체데이터사용 훈련
  
            4.한국어데이터에서 훈련
  
  
        


참고:
  https://github.com/Adamdad/keras-YOLOv3-mobilenet

  https://textvqa.org/textocr/dataset

  https://github.com/mvoelk/ssd_detectors

  https://wiserloner.tistory.com/
