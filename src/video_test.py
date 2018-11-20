#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

def bf_match(original_image):
    after_image = original_image

    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)

    #特徴点抽出
    kp1, des1 = detector.detectAndCompute(gray_image, None)
    kp2, des2 = detector.detectAndCompute(temp_gray_image, None)

    if des1 is None or des2 is None:
        return after_image

    #マッチング
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    #信頼性の高いマッチング結果のみから特徴点の座標を記録
    good_matches = []
    input_image_pts = []
    temp_image_pts = []
    for m in matches:
        if m.distance < 100.0: #信頼性が高いかのチェック
            good_matches.append(m)
            input_image_pts.append(map(int, kp1[m.queryIdx].pt))
            temp_image_pts.append(map(int, kp2[m.trainIdx].pt))

    #マッチングした特徴点がテンプレート画像での第何象限にあるかを記録
    first = []
    second = []
    third = []
    forth = []
    for input_image_point, temp_image_point in zip(input_image_pts, temp_image_pts):
        #2点の色を比較して誤認識を排除
        input_image_color = after_image[input_image_point[1], input_image_point[0]]
        temp_image_color = temp_image[temp_image_point[1], temp_image_point[0]]
        color_sub = input_image_color - temp_image_color
        color_sub = color_sub.astype(np.int8)
        if np.linalg.norm(color_sub)/442 > 0.1: #黒と白のユークリッド距離が441.6
            continue

        #マッチングした点を象限で区別
        if temp_image_point[0] > temp_center[0] and temp_image_point[1] < temp_center[1]:
            first.append(input_image_point)
        if temp_image_point[0] < temp_center[0] and temp_image_point[1] < temp_center[1]:
            second.append(input_image_point)
        if temp_image_point[0] < temp_center[0] and temp_image_point[1] > temp_center[1]:
            third.append(input_image_point)
        if temp_image_point[0] > temp_center[0] and temp_image_point[1] > temp_center[1]:
            forth.append(input_image_point)

    #象限によって色を分けて特徴点を表示
    for point in first:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (0,0,0), -1)
    for point in second:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (0,0,255), -1)
    for point in third:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (0,255,0), -1)
    for point in forth:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (255,0,0), -1)

    #各象限の特徴点から代表点を見つける
    upper_right = calc_center(first)
    upper_left = calc_center(second)
    lower_left = calc_center(third)
    lower_right = calc_center(forth)

    #代表点があればその点に沿って線を描画
    if upper_right is not None and upper_left is not None and lower_left is not None and lower_right is not None:
        cv2.line(after_image, (upper_right[0], upper_right[1]), (upper_left[0], upper_left[1]), (255,0,0), 10)
        cv2.line(after_image, (upper_left[0], upper_left[1]), (lower_left[0], lower_left[1]), (255,0,0), 10)
        cv2.line(after_image, (lower_left[0], lower_left[1]), (lower_right[0], lower_right[1]), (255,0,0), 10)
        cv2.line(after_image, (lower_right[0], lower_right[1]), (upper_right[0], upper_right[1]), (255,0,0), 10)

        points = Float32MultiArray()
        points.data = [upper_right[0], upper_right[1], upper_left[0], upper_left[1], lower_left[0], lower_left[1], lower_right[0], lower_right[1]]
        points_pub.publish(points)

    #入力画像とテンプレート画像をつなげてマッチング結果と共に表示
    after_image = cv2.drawMatches(after_image, kp1, temp_image, kp2, good_matches, None, flags=2)
    cv2.putText(after_image, "Drone Height = "+str(drone_height), (50, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),thickness=3)

    return after_image

#pointsの重心を出す
def calc_center(points):
    if len(points) == 0:
        return None
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    center = [x/len(points), y/len(points)]
    return center

def tempmatch(original_image):
    after_image = original_image

    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)

    result = cv2.matchTemplate(gray_image, temp_gray_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + temp_image.shape[1], top_left[1] + temp_image.shape[0])
    cv2.rectangle(after_image,top_left, bottom_right, 255, 2)
    return after_image

if __name__ == '__main__':
    rospy.init_node('marker_detect_node', anonymous=True)
    image_pub = rospy.Publisher("marker_detect/image_raw", Image, queue_size=1)
    points_pub = rospy.Publisher("marker_detect/points", Float32MultiArray, queue_size=1)
    bridge = CvBridge()

    cap = cv2.VideoCapture(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/video/1.mp4")
    temp_image = cv2.imread(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/temp1_cut.jpg") #第2引数が0でグレースケールで読み込むという意味
    temp_gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
    temp_center = [temp_image.shape[1]/2, temp_image.shape[0]/2]
    drone_height = 0.0

    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        #after_image = bf_match(frame)
        after_image = tempmatch(frame)
        image_pub.publish(bridge.cv2_to_imgmsg(after_image, "bgr8"))