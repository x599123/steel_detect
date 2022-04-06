# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, crop_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
import time, os
import requests
import json
from flask import Flask, render_template, Response
from threading import Thread

temp1 = 0
temp2 = 0
stat_list_sta = []
stat_list_cor = []
export_frame_in = []
export_frame_out = []
median_list = []
median_list1 = []
loaded_list = []
save_switch = True
voting_switch = False
web_out = []
contour_threshold = 150
alpha = 1.0
beta = 0
gamma = 1.0


@torch.no_grad()
def point_get_cor(ends, xy):  # 畫線
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    a0, a1 = np.abs(np.diff([ends[0], xy], axis=0))[0]
    if d0 < d1:
        slope = d0 / d1
        if ends[0, 0] > ends[1, 0]:
            return round(ends[0, 0] - (a1 * slope)), ends[0, 1] + a1
        else:
            return round(ends[0, 0] + (a1 * slope)), ends[0, 1] + a1


def web_view():  # Flask功能
    app = Flask(__name__)
    global web_out, web_in, save_switch

    def get_image():  # 即時影像 影像處理
        while True:
            jpeg = cv2.resize(web_out, (400, 490), interpolation=cv2.INTER_NEAREST)  # resize 圖片400*490
            ret, jpeg = cv2.imencode('.jpg', jpeg, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 壓縮圖片至jpg 品質設定50%
            jpeg = jpeg.tobytes()  # 轉換圖片至bytecode
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')  # return 圖片(generator型式)
            time.sleep(0.03)  # my Firefox needs some time to display image / Chrome displays image without it

    @app.route("/")
    def stream():  # 即時影像
        return Response(get_image(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def one_image(wb):
        jpeg = cv2.resize(wb, (400, 490), interpolation=cv2.INTER_NEAREST)  # resize 圖片400*490
        ret, jpeg = cv2.imencode('.jpg', jpeg)  # 壓縮圖片至jpg 品質設定50%
        jpeg = jpeg.tobytes()  # 轉換圖片至bytecode
        return jpeg  # return 圖片

    @app.route("/out")
    def out():  # return 標記後的圖片
        return Response(one_image(web_out), mimetype="image/jpeg")

    @app.route("/in")
    def webin():  # return 標記前的圖片
        return Response(one_image(web_in), mimetype="image/jpeg")

    @app.route("/shutdown")
    def shutdown():  # 關閉程式
        os.system(
            'ps -ef | grep detect_steel_light_stream.py| grep -v grep | awk \'{print $2}\' | xargs kill -9')  # kill 自己本身
        return Response('shutdown')

    @app.route("/savestatus")
    def savestatus():  # 回傳存檔狀態
        global save_switch
        # exit()
        return Response(str(save_switch))  # 回傳存檔狀態

    @app.route("/saveon")
    def saveon():
        global save_switch
        # exit()
        save_switch = True  # 存檔開關設開
        return Response(str(save_switch))  # 回傳存檔狀態

    @app.route("/saveoff")
    def saveoff():
        global save_switch
        # exit()
        save_switch = False  # 存檔開關設關
        return Response(str(save_switch))  # 回傳存檔狀態

    app.run('0.0.0.0')


def detect_steel_opencv(y1, img_ori, img):  # 畫線算出偏移量
    global contour_threshold, alpha, beta, gamma
    try:
        res = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  # 補光alpha beta
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        res = cv2.LUT(res, lookUpTable)
        opencv_start = time.process_time()

        # cv2.imwrite('opencv_img.jpg',img)
        img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # 圖片轉灰階
        thresh = cv2.threshold(img, contour_threshold, 255, cv2.THRESH_BINARY)[1]  # 二值化 去背
        # cv2.imwrite('opencv_thresold.jpg', thresh)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        big_contour = max(contours, key=cv2.contourArea)
        big_contour = cv2.UMat.get(big_contour)

        # print(big_contour)

        img = cv2.UMat.get(img)
        result = np.zeros_like(img)
        # cv2.imwrite('opencv_result.jpg', result)
        # print(big_contour.min())
        cv2.drawContours(result, [big_contour], 0, (255, 255, 255), cv2.FILLED)
        # cv2.imwrite('opencv_big_contour.jpg', result)
        without_back = cv2.bitwise_and(img, result)
        # cv2.imwrite('opencv_bitwise_and.jpg', without_back)
        img_raw = cv2.cvtColor(without_back, cv2.COLOR_GRAY2BGR)
        x_y_df = pd.DataFrame(big_contour[:, 0, :], columns=["x", "y"])
        head_df = x_y_df['x'].loc[x_y_df['y'] <= x_y_df["y"].min() + 2]
        tail_df = x_y_df['x'].loc[x_y_df['y'] >= x_y_df["y"].max() - 5]
        points = np.array(
            [[head_df.max(), x_y_df["y"].min()], [tail_df.max(), x_y_df["y"].max()], [tail_df.min(), x_y_df["y"].max()],
             [head_df.min(), x_y_df["y"].min()]], np.int32)
        right_cor_area = np.array([points[0].reshape(2), points[1].reshape(2)])
        mid_cor_area = np.array([[round((head_df.max() + head_df.min()) / 2), x_y_df["y"].min()],
                                 [round((tail_df.max() + tail_df.min()) / 2), x_y_df["y"].max()]])
        abstract_list = []
        cor_list = []
        x_y_df = x_y_df.loc[x_y_df["y"] < (x_y_df["y"].max() - (x_y_df["y"].max() - x_y_df["y"].min()) / 4)]
        contour_rights = np.array(x_y_df.sort_values(['x'], ascending=False).drop_duplicates(
            subset=['y']).sort_values(['y']))
        for contour_right in reversed(contour_rights):
            right_cordinate = point_get_cor(right_cor_area, contour_right)
            mid_cordinate = point_get_cor(mid_cor_area, contour_right)
            # print(contour_right[0],mid_cordinate[0])
            if contour_right[0] > mid_cordinate[0] and contour_right[0] - mid_cordinate[0] > 12:
                # print(contour_right[0], mid_cordinate[0])
                cv2.circle(img_raw, tuple(mid_cordinate), 1, (0, 0, 255), 1)
                cv2.circle(img_raw, tuple(contour_right), 1, (0, 0, 255), 1)
                abstract_list.append(right_cordinate[0] - contour_right[0])
                cor_list.append(contour_right)
        points = points.reshape((-1, 1, 2))
        cv2.line(img_raw, tuple(points[0].reshape(2)), tuple(points[1].reshape(2)), (0, 255, 0), 1)

        cv2.polylines(img_raw, pts=[points], isClosed=False, color=(0, 0, 255), thickness=1)
        cv2.polylines(img_ori, pts=[points], isClosed=False, color=(0, 0, 255), thickness=1)
        # cv2.imwrite('opencv_img_raw.jpg', img_raw)
        # cv2.imwrite('opencv_img_ori.jpg', img_ori)

        # print(pathtowrite+"_contours.jpg")

        if len(abstract_list) > 0:
            return_value = max(abstract_list)
            cv2.putText(img_raw, str(sum(abstract_list) / len(abstract_list)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_raw, str(max(abstract_list)), tuple(cor_list[abstract_list.index(max(abstract_list))]),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_ori, str(max(abstract_list)), tuple(cor_list[abstract_list.index(max(abstract_list))]),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
            # (pathtowrite + "_contours.jpg", img_raw)
            opencv_end = time.process_time()
            print("opencv執行時間單跟：%f 秒" % (opencv_end - opencv_start))
            # cv2.imwrite('opencv_img_raw_num.jpg', img_raw)
            # cv2.imwrite('opencv_img_ori_num.jpg', img_ori)
            # cv2.imshow("filter",img_raw)
            # cv2.imshow("filter_ori",img_ori)
            # cv2.waitKey(1)
            steel_detect_result = return_value, img_raw
            return return_value, img_raw
        # (pathtowrite + "_contours.jpg", img_raw)
        opencv_end = time.process_time()
        print("opencv執行時間：%f 秒" % (opencv_end - opencv_start))
    except:
        pass


def run(weights='best.pt',  # model.pt path(s)
        source='1203/[CH01]2021-12-02-15.00.04.mp4',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.55,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        tfl_int8=False,  # INT8 quantized TFLite model
        ):
    save_img = not nosave and source.endswith('.mp4')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = True
    batch_num = 0
    total_curve_quantity = 0
    total_ok_quntity = 0
    voting_switch_list = []
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader 取得即時影像
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    global temp1, temp2, stat_list_sta, stat_list_cor, export_frame_in, export_frame_out, web_in, web_out, save_switch, loaded_list, voting_switch

    enum = 0

    start = time.process_time()
    # loadstream self.sources, img, img0, img_raw, img_ori, None
    for path, img, im0s, imoris, img_oris, vid_cap in dataset:  # img是要餵給yolo辨識的預處理影像 img_oris為未裁切原圖
        curve_steel_quntity = 0
        enum = enum + 1
        web_in = im0s[0]  # im0s為透視轉換圖
        web_out = imoris[0]  # imoris為原圖裁切
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:  # yolo推論 將圖片餵進AI模型
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if tfl_int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if tfl_int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)

        # NMS 預測結果處理 找到最佳框
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions

        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0, imori, img_ori, frame = path[i], f'{i}: ', im0s[i].copy(), imoris[i].copy(), img_oris[
                    i].copy(), dataset.count
                # im0為透視轉換圖
                # imori為原圖裁切
                # img_ori為原圖
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # cv2.imshow('imoris', img_ori)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                onum = 0
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                abstract_list = []

                for *xyxy, conf, cls in reversed(det):
                    # 對yolo框出的每根鋼條做檢測演算
                    list_xyxy = torch.tensor(xyxy).view(-1, 4)  # tensor轉list
                    if save_img or save_crop or view_img:  # Add bbox to image
                        if int(list_xyxy[:, 3]) > 800 and int(list_xyxy[:, 1]) < 710 and int(list_xyxy[:, 1]) > 430:
                            steel_detect_result = 0
                            onum = onum + 1
                            detect_run = True
                            crop_img = crop_one_box(xyxy, im0, file=save_dir / 'crops' / f'{p.stem}_{enum}_{onum}.jpg',
                                                    BGR=True, )
                            # yolo框出鋼條取出單根影像
                            # cv2.imwrite('opencv_imori.jpg', imori)
                            steel_detect_result = detect_steel_opencv(int(list_xyxy[:, 1]), imori, crop_img)
                            # 單根鋼條餵入彎直檢測演算法
                            web_out = imori
                            # cv2.imshow('aav',steel_detect_result[1])
                            # cv2.waitKey(1)
                            if steel_detect_result:
                                abstract_list.append(
                                    [int(list_xyxy[:, 0]), int(list_xyxy[:, 1]), steel_detect_result[0]])
                                # 紀錄座標、鋼條偏移量、投票trigger
                            if int(list_xyxy[:, 1]) > 430:
                                voting_switch_list.append(int(list_xyxy[:, 1]))
                                # 離開辨識區累積投票trigger
                                # print(voting_switch_list)
                cv2.imshow('img_ori', imori)
            try:
                if abstract_list:
                    loaded_list.append(abstract_list)
                    # 儲存 紀錄座標、鋼條偏移量、投票trigger的list 例如若有五根鋼條abstract_list會累積五個
                if len(loaded_list) <= 23:
                    steel_img_with_line = imori
                    steel_img = img_ori
                    # 投票完會儲存照片，開始投票時圖片已經更新 因此要保留投票區間的圖片
            except:
                pass
            # print(len(voting_switch_list))

            # print("loaded_list len:", len(loaded_list))
            # print("voting_switch_list len:", len(voting_switch_list))
            if len(voting_switch_list) >= 20 and len(loaded_list) > 23 and len(abstract_list) == 0:
                # 當在辨識區trigger 離開辨識區trigger 滿足一批的數量 加上當前是離開辨識區的狀況 就會開始投票
                batch_num = batch_num + 1

                voting_start = time.process_time()
                len_list = []

                for q in loaded_list:
                    # print(len(i))
                    len_list.append(len(q))
                # 計算每個frame有幾根鋼條
                for h in loaded_list:

                    if len(h) == max(set(len_list), key=len_list.count):
                        # 找出第一個鋼條數量最常出現的frame 例如這批鋼條五根 找出第一個有五根鋼條的frame
                        first_list = sorted(h, key=lambda s: s[0])
                        break
                voting_list = []
                for list_line in loaded_list:
                    temp_list = [0] * len(first_list)  # 建立狀態list [0,0,0,0,0]
                    bad_contours_list = [0] * len(first_list)  # 建立狀態list [0,0,0,0,0]
                    list_line = sorted(list_line, key=lambda s: s[0])  # 依照y做排序
                    try:
                        for g, j in enumerate(list_line):
                            if abs(j[0] - first_list[g][0]) > 50:  # 取得鋼條垂直巨大於50要特別小心 常常框錯
                                # j[0] y j[1] x j[2] bias
                                for num, first_list_x in enumerate(first_list):
                                    # print(j[0], first_list_x[0])
                                    if abs(j[0] - first_list_x[0]) < 50:  # 取得鋼條水平距離小於50
                                        temp_list[num] = [j[2], j[1]]
                            else:
                                temp_list[g] = [j[2], j[1]]  # 轉存鋼鐵偏移量 鋼鐵x座標
                    except:
                        pass
                    voting_list.append(temp_list)
                # voting_list{[bias,y1]}
                temp_list = [0] * len(first_list)  # 每個位置鋼條狀態 彎的數量
                temp_list_total = [0] * len(first_list)  # 每個位置總共鋼條數
                temp_list_bias = [0] * len(first_list)  # 每個位置鋼條偏移量
                # print(voting_list)
                for voting_line in voting_list:
                    # print(voting_line)
                    for steel_number, voting_num in enumerate(voting_line):
                        if type(voting_num) != int and voting_num[0] < 12:
                            temp_list_total[steel_number] = temp_list_total[steel_number] + 1
                            if voting_num[0] > temp_list_bias[steel_number]:
                                temp_list_bias[steel_number] = voting_num[0]
                            if voting_num[0] >= 6 and voting_num[1] > 670:
                                # print(steel_number + 1, voting_num[0])
                                temp_list[steel_number] = temp_list[steel_number] + 1
                            elif voting_num[0] >= 7:
                                # print(steel_number + 1, voting_num[0])
                                temp_list[steel_number] = temp_list[steel_number] + 1
                        else:
                            bad_contours_list[steel_number] = bad_contours_list[steel_number] + 1
                steel_detail = ''
                send_steel_status = ''
                folder_name = ''
                for steel_number, curve_quantity in enumerate(temp_list):

                    if curve_quantity > temp_list_total[steel_number] / 20:
                        # print(steel_number + 1, "cu", temp_list_bias[steel_number], curve_quantity,
                        #       temp_list_total[steel_number])
                        curve_steel_quntity = curve_steel_quntity + 1  # log記錄一批幾隻彎的鐵
                        total_curve_quantity = total_curve_quantity + 1  # log記錄從程式開始至今幾隻彎的鐵
                        steel_detail = f'{steel_detail}_{steel_number + 1}_cu_{temp_list_bias[steel_number]}'  # 檔名製造
                        send_steel_status = send_steel_status + '1'
                        folder_name = folder_name + "_" + str(steel_number + 1) + "th" + "_" + "cu"
                    else:
                        steel_detail = f'{steel_detail}_{steel_number + 1}_ok_{temp_list_bias[steel_number]}'
                        total_ok_quntity = total_ok_quntity + 1  # log記錄一批幾隻好的鐵
                        send_steel_status = send_steel_status + '0'  # log記錄從程式開始至今幾隻好的鐵
                        folder_name = folder_name + "_" + str(steel_number + 1) + "th" + "_" + "ok"  # 檔名製造
                localtime = str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
                end = time.process_time()
                steel_log = "source: " + str(source.split("\\")[-1]).strip(".mp4") + '\n'
                steel_log = steel_log + "How_many_steel_in_this_batch: " + str(len(temp_list_total)) + '\n'
                steel_log = steel_log + "yolo_detection_quntity: " + str(len(loaded_list)) + '\n'
                steel_log = steel_log + "steel_quantity_list: " + str(temp_list_total) + '\n'
                steel_log = steel_log + "curve_list: " + str(temp_list) + '\n'
                steel_log = steel_log + "ok_list: " + str(
                    [temp_list_total[i] - temp_list[i] for i in range(len(temp_list_total))]) + '\n'
                steel_log = steel_log + "bad_contours_list: " + str(bad_contours_list) + '\n'
                steel_log = steel_log + "curve_steel_quntity:" + str(curve_steel_quntity) + '\n'
                steel_log = steel_log + "steel_detail:" + steel_detail + '\n'
                steel_log = steel_log + "cost_time_batch：%f s" % (end - start) + '\n'
                steel_log = steel_log + f'Filename: {localtime}{steel_detail}_{alpha}_{beta}_{gamma}_{contour_threshold}_withlines.jpg' + '\n'
                steel_log = steel_log + f"Alpha: {alpha} Beat:{beta} Gamma :{gamma} Thresold: {contour_threshold}" + '\n'
                steel_log = steel_log + "------------------------------------" + '\n'
                data = {"Total_photo": 0,
                        "Use_photo": 0,
                        "Total_steel": str(len(temp_list_total)),
                        "Line_steel": len(temp_list_total) - curve_steel_quntity,
                        "Curve_steel": curve_steel_quntity,
                        "Identify_time": localtime,
                        "Steel": send_steel_status}
                # req = requests.post('http://127.0.0.1:8080/steel/save_steel_log/', data=json.dumps(data), verify=False)
                # print(req.status_code)
                # print(req.json())
                # 傳出json post
                folder1 = "./steel_detect_n/outimage/" + localtime + folder_name
                # 儲存log
                with open(r'statics.txt', 'a+') as f:
                    f.write(steel_log + '\n')
                if curve_steel_quntity == 4:
                    with open(r'statics_try_argument.txt', 'a+') as f:
                        f.write(steel_log + '\n')
                if curve_steel_quntity >= 1:
                    with open(r'statics_try_if_curve.txt', 'a+') as f:
                        f.write(steel_log + '\n')
                print(steel_log)
                print(f'{save_dir}{localtime}{steel_detail}')
                # 建立狀態以及時間資料夾
                if not os.path.exists(folder1):
                    os.mkdir(folder1)
                if save_switch:
                    cv2.imwrite(
                        f'{folder1}/{localtime}{steel_detail}_{alpha}_{beta}_{gamma}_{contour_threshold}_withlines.jpg',
                        steel_img_with_line)
                    cv2.imwrite(f'{folder1}/{localtime}{steel_detail}_raw.jpg', steel_img)

                loaded_list = []  # 復歸紀錄list
                voting_switch_list = []  # 復歸投票紀錄list
                voting_end = time.process_time()
                print("voting_cost：%f 秒" % (voting_end - voting_start))
                start = time.process_time()

            end = time.process_time()
            print(enum)
            print("執行時間：%f 秒" % (end - start))

            print(f'{s}Done. ({t2 - t1:.3f}s)')
            ## Save results (image with detections)

    print(f'Done. ({time.time() - t0:.3f}s)')
    # print('Thread')
    with open(r'statics_total.txt', 'a+') as f:
        f.write(f"total imference frame {enum} \n")
        f.write(f"ok {total_ok_quntity} \n")
        f.write(f"curve{total_curve_quantity}\n")
    # 紀錄影片辨識結束的log


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='steel_detect_n/20220311.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='steel_detect_n/03101453_reverse-.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--tfl-int8', action='store_true', help='INT8 quantized TFLite model')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     contour_threshold = 150
#     t1 = Thread(target=web_view, args=())
#     t1.start()
#     main(opt)

class steel_detect:
    def __init__(self):
        opt = parse_opt()

        t1 = Thread(target=web_view, args=())
        t1.start()
        main(opt)
