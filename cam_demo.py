import os
import sys
from glob import glob
import time
import argparse
import cv2
import torch


from facedetection.detect import FaceDetector
from lmdetection.detect import LMDetector
from calc_lm import face_metrics

TIME_TH = 3
FACE_TH = 0.6
EAR_TH = 0.2 # 자긍면 문제
HR_TH = (0.2, 0.95) # 이 구간 문제
SR_TH = (0.5, 2) # 이 구간 밖 문제


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fd_model = FaceDetector(device)
    lm_model = LMDetector(device)


    if os.path.isfile(args.img_path):
        img_list = [args.img_path]
    elif os.path.isdir(args.img_path):
        img_list = sorted(glob(os.path.join(args.img_path, '**', '*'), recursive=True))

    print(img_list)


    ear_status = False
    head_status = False
    face_status = False

    ear_status_warning = False
    head_status_warning = False
    face_status_warning = False

    ear_status_time = 0
    head_status_time = 0
    face_status_time = 0

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        face_dets= fd_model.detection(img)
        cur_time = time.time()
        # 전방 미주시
        if len(face_dets) == 0 and face_dets[0][4] < FACE_TH:
            if not face_status:
                face_status_time = cur_time
                face_status = True
            
            if face_status and (cur_time - face_status_time) > TIME_TH:
                face_status_warning = True
                if face_status_warning:
                    cv2.rectangle(img, (0,0), (300, 100), (255,0,0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, 'Warning!', (5,70),cv2.FONT_HERSHEY_DUPLEX, 2,(255,255,255), thickness=3, lineType=cv2.LINE_AA)
            continue
        
        face_status = False
        face_status_warning = False

        pts = lm_model.detection(img, face_dets[0])
        EAR, HR, SR = face_metrics(pts)

        # EAR
        if EAR < EAR_TH:
            cur_time = time.time()
            if not ear_status:
                ear_status_time = cur_time
                ear_status = True
            if ear_status and cur_time- ear_status_time > TIME_TH:
                ear_status_warning = True

        else:
            ear_status = False
            ear_status_warning = False

        # 고개 이상 확인 
        if (HR_TH[0] < HR and HR_TH[1] > HR) or (SR < SR_TH[0] and SR_TH[1] < SR):
            cur_time = time.time()
            if not head_status:
                head_status_time = cur_time
                head_status = True

            if head_status and (cur_time - head_status_time) > TIME_TH:
                head_status_warning = True
        else:
            head_status = False
            head_status_warning = False

        if ear_status_warning or head_status_warning:
            print(head_status_warning, ear_status_warning)
            cv2.putText(img, 'Warning!', (300,70),cv2.FONT_HERSHEY_DUPLEX, 2,(255,255,255), thickness=3, lineType=cv2.LINE_AA)


        cv2.putText(img, f'EAR:{EAR:.2f} ', (0,100),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(img, f'SR:{SR:.2f}', (0,150),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(img, f'HR:{HR:.2f}', (0,200),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0), thickness=3, lineType=cv2.LINE_AA)
        cv2.imshow("VideoFrame", img)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()