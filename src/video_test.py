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

"""
def tempmatch(original_image):
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    after_image = original_image

    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)

    for method in methods:
        method = eval(method)
        result = cv2.matchTemplate(gray_image, temp_gray_image, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + temp_image.shape[1], top_left[1] + temp_image.shape[0])
        cv2.rectangle(after_image, top_left, bottom_right, (255,0,0), 2)

    return after_image
"""

"""
def labeling(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)[1]

    label = cv2.connectedComponentsWithStats(binary_image)
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)

    for i in range(n):
        # 各オブジェクトの外接矩形を表示
        if data[i][4] < 100:
            continue
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
   
        cut_image = gray_image[y0-5:y1+5, x0-5:x1+5] #マーカ構成要素部分を切り出し
        edges = cv2.Canny(cut_image,100,200)
        contours= cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
        for contour in contours:
            #輪郭を直線近似
            arclen = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * arclen, True)
            #cv2.drawContours(cut_image, approx, -1, (0,255,0), 3)
            if len(approx) <= 3:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (0,0,255), -1)
            elif len(approx) == 4:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (0,255,0), -1)
            elif len(approx) >= 5:
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (255,0,0), -1)
        
        #cv2.imshow(str(i),cut_image)
        #cv2.waitKey(1)

    return image
"""

"""
def color_labeling(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_g = np.array([60, 50, 50])
    upper_g = np.array([95, 255, 255])
    lower_r = np.array([160, 50, 50])
    upper_r = np.array([180, 255, 255])
    lower_b = np.array([98, 50, 230])
    upper_b = np.array([100, 255, 255])
    lower_black = np.array([100, 50, 50])
    upper_black = np.array([130, 255, 120])

    img_mask = cv2.inRange(hsv, lower_g, upper_g)

    contours= cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    areas = np.array(list(map(cv2.contourArea, contours)))

    if len(areas) == 0:
        return image

    max_idx = np.argmax(areas)
    #max_area = areas[max_idx]
    result = cv2.moments(contours[max_idx])
    if result["m00"] != 0:
        x = int(result["m10"]/result["m00"])
        y = int(result["m01"]/result["m00"])
        cv2.circle(image, (x, y), 5, (255,0,0), -1)

    return image
"""

def bf_match(original_image):
    after_image = original_image

    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)

    #特徴点抽出
    kp_input, des_input = detector.detectAndCompute(gray_image, None)
    
    if des_input is None or des_temp is None:
        return after_image

    #マッチング
    matches = bf.match(des_input,des_temp)
    matches = sorted(matches, key = lambda x:x.distance)

    #信頼性の高いマッチング結果のみから特徴点の座標を記録
    good_matches = []
    input_image_pts = []
    temp_image_pts = []
    for m in matches:
        if m.distance < 60.0: #信頼性が高いかのチェック
            good_matches.append(m)
            input_image_pts.append(map(int, kp_input[m.queryIdx].pt))
            temp_image_pts.append(map(int, kp_temp[m.trainIdx].pt))

    #マッチングした特徴点がテンプレート画像での第何象限にあるかを記録
    first = []
    second = []
    third = []
    forth = []
    fifth = []
    sixth = [] 
    for input_image_point, temp_image_point in zip(input_image_pts, temp_image_pts):
        #2点の色を比較して誤認識を排除
        input_image_color = after_image[input_image_point[1], input_image_point[0]]
        temp_image_color = temp_image[temp_image_point[1], temp_image_point[0]]
        color_sub = input_image_color - temp_image_color
        color_sub = color_sub.astype(np.int8)
        if np.linalg.norm(color_sub)/441.6 > 0.1: #黒と白のユークリッド距離が441.6
            continue

        """
        #マッチングした点を象限で区別 4点版
        if temp_image_point[0] > temp_center[0] and temp_image_point[1] < temp_center[1]:
            first.append(input_image_point)
        if temp_image_point[0] < temp_center[0] and temp_image_point[1] < temp_center[1]:
            second.append(input_image_point)
        if temp_image_point[0] < temp_center[0] and temp_image_point[1] > temp_center[1]:
            third.append(input_image_point)
        if temp_image_point[0] > temp_center[0] and temp_image_point[1] > temp_center[1]:
            forth.append(input_image_point)
        """

        #マッチングした点を象限で区別 6点版
        if temp_image_point[0] > temp_center[0] and temp_image_point[1] < temp_height/3:
            first.append(input_image_point)
        if temp_image_point[0] < temp_center[0] and temp_image_point[1] < temp_height/3:
            second.append(input_image_point)
        if temp_image_point[0] < temp_center[0] and temp_image_point[1] > 2*(temp_height/3):
            third.append(input_image_point)
        if temp_image_point[0] > temp_center[0] and temp_image_point[1] > 2*(temp_height/3):
            forth.append(input_image_point)
        if temp_image_point[0] < temp_center[0] and (temp_image_point[1] > temp_height/3 and temp_image_point[1] < 2*(temp_height/3)):
            fifth.append(input_image_point)
        if temp_image_point[0] > temp_center[0] and (temp_image_point[1] > temp_height/3 and temp_image_point[1] < 2*(temp_height/3)):
            sixth.append(input_image_point)

    #象限によって色を分けて特徴点を表示
    """
    for point in first:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (0,0,0), -1)
    for point in second:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (0,0,255), -1)
    for point in third:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (0,255,0), -1)
    for point in forth:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (255,0,0), -1)
    for point in fifth:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (255,0,255), -1)
    for point in sixth:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (0,255,255), -1)
    """

    #各象限の特徴点から代表点を見つける
    upper_right = calc_center(first)
    upper_left = calc_center(second)
    lower_left = calc_center(third)
    lower_right = calc_center(forth)
    middle_right = calc_center(fifth)
    middle_left = calc_center(sixth)

    #代表点があればその点に沿って線を描画
    if upper_right is not None and upper_left is not None and lower_left is not None and lower_right is not None and middle_right is not None and middle_left is not None:
        #cv2.line(after_image, (upper_right[0], upper_right[1]), (upper_left[0], upper_left[1]), (255,0,0), 5)
        #cv2.line(after_image, (upper_left[0], upper_left[1]), (lower_left[0], lower_left[1]), (255,0,0), 5)
        #cv2.line(after_image, (lower_left[0], lower_left[1]), (lower_right[0], lower_right[1]), (255,0,0), 5)
        #cv2.line(after_image, (lower_right[0], lower_right[1]), (upper_right[0], upper_right[1]), (255,0,0), 5)
        cv2.circle(after_image, (upper_right[0], upper_right[1]), 7, (0,0,0), -1)
        cv2.circle(after_image, (upper_left[0], upper_left[1]), 7, (0,0,255), -1)
        cv2.circle(after_image, (lower_left[0], lower_left[1]), 7, (0,255,0), -1)
        cv2.circle(after_image, (lower_right[0], lower_right[1]), 7, (255,0,0), -1)
        cv2.circle(after_image, (middle_right[0], middle_right[1]), 7, (255,0,255), -1)
        cv2.circle(after_image, (middle_left[0], middle_left[1]), 7, (0,255,255), -1)

        points = Float32MultiArray()
        points.data = [upper_right[0], upper_right[1], upper_left[0], upper_left[1], lower_left[0], lower_left[1], lower_right[0], lower_right[1], middle_left[0], middle_left[1], middle_right[0], middle_right[1]]
        points_pub.publish(points)

    #入力画像とテンプレート画像をつなげてマッチング結果と共に表示
    after_image = cv2.drawMatches(after_image, kp_input, temp_image, kp_temp, good_matches, None, flags=2)

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

if __name__ == '__main__':
    rospy.init_node('marker_detect_node', anonymous=True)
    image_pub = rospy.Publisher("marker_detect/image_raw", Image, queue_size=1)
    points_pub = rospy.Publisher("marker_detect/points", Float32MultiArray, queue_size=1)
    bridge = CvBridge()

    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    cap = cv2.VideoCapture(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/pattern1/1_1.mp4")

    #特徴点マッチングのときのみ使用
    temp_image = cv2.imread(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/pattern1/temp.jpg")
    temp_gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
    temp_center = [temp_image.shape[1]/2, temp_image.shape[0]/2]
    temp_height = temp_image.shape[0]

    #テンプレート画像の特徴点を抽出
    kp_temp, des_temp = detector.detectAndCompute(temp_gray_image, None)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        after_image = bf_match(frame)
        image_pub.publish(bridge.cv2_to_imgmsg(after_image, "bgr8"))