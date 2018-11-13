#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image = bf_match(cv_image)
    image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))

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

    #マッチングした特徴点の座標を記録
    input_image_pts = [list(map(int, kp1[m.queryIdx].pt)) for m in matches[:10]]
    temp_image_pts = [list(map(int, kp2[m.trainIdx].pt)) for m in matches[:10]]
    #print "%d %d" % (len(input_image_pts),len(temp_image_pts))

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
        if np.linalg.norm(color_sub)/442 > 0.2: #黒と白のユークリッド距離が441.6
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
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (255,0,0), -1)
    for point in second:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (0,255,0), -1)
    for point in third:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (0,0,255), -1)
    for point in forth:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 10, (0,0,0), -1)

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

    #入力画像とテンプレート画像をつなげてマッチング結果と共に表示
    after_image = cv2.drawMatches(after_image, kp1, temp_image, kp2, matches[:10], None, flags=2)

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
    
    bridge = CvBridge()
    temp_image = cv2.imread(os.environ["HOME"]+"/bebop_ws/src/drone_marker_pkg/resource/temp1_50.jpg") #第2引数が0でグレースケールで読み込むという意味
    temp_gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
    temp_center = [temp_image.shape[1]/2, temp_image.shape[0]/2]
    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    rospy.Subscriber("bebop/image_raw", Image, callback)
    rospy.spin()

"""
def bf_match(original_image):
    after_image = original_image

    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)
    kp1, des1 = detector.detectAndCompute(gray_image, None)
    kp2, des2 = detector.detectAndCompute(temp_image, None)

    if des1 is None or des2 is None:
        return after_image

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    points = []
    for match in matches[:10]:
        if match.trainIdx < len(kp1):
            points.append(kp1[match.trainIdx].pt)

    for point in points:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (0,0,255), -1)

    upper_left, lower_right = find_end_point(points)
    if upper_left is not None and lower_right is not None:
        after_image = cv2.rectangle(after_image, (int(upper_left[0]),int(upper_left[1])), (int(lower_right[0]),int(lower_right[1])), (255,0,0), 5)
        after_image = cv2.circle(after_image, (int(upper_left[0]), int(upper_left[1])), 10, (255,0,0), -1)
        after_image = cv2.circle(after_image, (int(lower_right[0]), int(lower_right[1])), 10, (255,0,0), -1)
    after_image = cv2.drawMatches(original_image, kp1, temp_image, kp2, matches[:10], None, flags=2)

    return after_image

#マッチングした特徴点から左上と右下の点を見つける
def find_end_point(points):
    upper_left = None
    lower_right = None
    for point in points:
        if upper_left is None:
            upper_left = point
        else:
            if point[0] < upper_left[0] and point[1] < upper_left[1]:
                upper_left = point

        if lower_right is None:
            lower_right = point
        else:
            if point[0] > lower_right[0] and point[1] > lower_right[1]:
                lower_right = point
    return upper_left,lower_right
"""

"""
def featureMatching(original_image):
    kp1, des1 = detector.detectAndCompute(original_image, None)
    kp2, des2 = detector.detectAndCompute(temp_image, None)

    if des1 is None or des2 is None:
        return original_image

    matches = bf.knnMatch(des1, des2, k=2)
    good_points = []
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            if(m.trainIdx < len(kp1)):
                good_points.append(kp1[m.trainIdx].pt)

    #if(len(good_points) == 0):
    #    return original_image

    #特徴点のある点に赤丸を描く
    #x_sum = 0.0
    #y_sum = 0.0
    #points_sum = 0
    for point in good_points:
        #color = original_image[point[1],point[0]]
        #green = original_image.item(point[1],point[0],1)
        #if(green > 40):
        #x_sum += point[0]
        #y_sum += point[1]
        #points_sum += 1
        after_image = cv2.circle(original_image, (int(point[0]), int(point[1])), 5, (0,0,255), -1)
    return after_image

    #if(points_sum != 0):
    #    after_image = cv2.circle(original_image, (int(x_sum/points_sum), int(y_sum/points_sum)), 5, (0,0,255), -1)
    #    return after_image

    #after_image = cv2.drawMatchesKnn(original_image, kp1, temp_image, kp2, good, None, flags=2)
    #return after_image
    #return original_image
"""

"""
def markerRecognition(original_image):
    after_image = original_image

    height, width, channels = after_image.shape
    image_size = height * width

    #グレースケール変換
    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)

    retval, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_TOZERO_INV )
    binary_image = cv2.bitwise_not(binary_image)
    retval, binary_image= cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary",binary_image)
    
    countours_image, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("countour", countours_image)

    cv2.waitKey(1)

    for contour in contours:
        area = cv2.contourArea(contour)

        #過度に大小なエリアは弾く
        if area > image_size * 0.4 and area < image_size * 0.01:
            continue

        #輪郭を直線近似
        arclen = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * arclen, True)

        #直線近似した輪郭を辺の数で場合分け
        if(len(approx) >= 3 and len(approx) <= 5):
            x,y,w,h = cv2.boundingRect(approx)
            #cv2.rectangle(after_image, (x,y), (x+w,y+h), (0,0,255), 3)
            cv2.circle(after_image, (x+w/2, y+h/2), 5, (0,0,255), -1)
            #cv2.drawContours(after_image, approx, -1, (0,0,255), 3)

    return after_image
"""