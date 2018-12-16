#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

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
    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    input_image = cv2.imread("../save/input_image.jpg")
    input_gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)

    temp_image = cv2.imread("../save/temp_image.jpg")
    temp_gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
    temp_center = [temp_image.shape[1]/2, temp_image.shape[0]/2]
    
    kp_input, des_input = detector.detectAndCompute(input_gray_image, None)
    kp_temp, des_temp = detector.detectAndCompute(temp_gray_image, None)

    input_key_image = cv2.drawKeypoints(input_image,kp_input) #被探索画像の特徴点
    temp_key_image = cv2.drawKeypoints(temp_image,kp_temp) #テンプレート画像の特徴点

    matches = bf.match(des_input,des_temp)
    matches = sorted(matches, key = lambda x:x.distance)

    result1 = cv2.drawMatches(input_image, kp_input, temp_image, kp_temp, matches, None, flags=2) #特徴点マッチングの結果

    cv2.imshow("a",temp_key_image)
    cv2.imshow("b",input_key_image)
    cv2.imshow("c",result1)
    cv2.imwrite("../save/a.jpg",temp_key_image)
    cv2.imwrite("../save/b.jpg",input_key_image)
    cv2.imwrite("../save/c.jpg",result1)

    good_matches = []
    input_image_pts = []
    temp_image_pts = []
    for m in matches:
        if m.distance < 60.0: #信頼性が高いかのチェック
            input_image_color = input_image[int(kp_input[m.queryIdx].pt[1]), int(kp_input[m.queryIdx].pt[0])]
            temp_image_color = temp_image[int(kp_temp[m.trainIdx].pt[1]), int(kp_temp[m.trainIdx].pt[0])]
            color_sub = input_image_color - temp_image_color
            color_sub = color_sub.astype(np.int8)
            if np.linalg.norm(color_sub)/441.6 <= 0.1: #黒と白のユークリッド距離が441.6
                good_matches.append(m)
                input_image_pts.append(map(int, kp_input[m.queryIdx].pt))
                temp_image_pts.append(map(int, kp_temp[m.trainIdx].pt))

    result2 = cv2.drawMatches(input_image, kp_input, temp_image, kp_temp, good_matches, None, flags=2) #距離と色による除外後

    cv2.imshow("d",result2)
    cv2.imwrite("../save/d.jpg",result2)

    #マッチングした特徴点がテンプレート画像での第何象限にあるかを記録
    first = []
    second = []
    third = []
    forth = []
    for input_image_point, temp_image_point in zip(input_image_pts, temp_image_pts):
        #2点の色を比較して誤認識を排除
        input_image_color = input_image[input_image_point[1], input_image_point[0]]
        temp_image_color = temp_image[temp_image_point[1], temp_image_point[0]]
        color_sub = input_image_color - temp_image_color
        color_sub = color_sub.astype(np.int8)
        if np.linalg.norm(color_sub)/441.6 > 0.1: #黒と白のユークリッド距離が441.6
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
        color_point_image = cv2.circle(input_image, (int(point[0]), int(point[1])), 5, (0,0,0), -1)
    for point in second:
        color_point_image = cv2.circle(color_point_image, (int(point[0]), int(point[1])), 5, (0,0,255), -1)
    for point in third:
        color_point_image = cv2.circle(color_point_image, (int(point[0]), int(point[1])), 5, (0,255,0), -1)
    for point in forth:
        color_point_image = cv2.circle(color_point_image, (int(point[0]), int(point[1])), 5, (255,0,0), -1)

    cv2.imshow("e",color_point_image)
    cv2.imwrite("../save/e.jpg",color_point_image)

    #各象限の特徴点から代表点を見つける
    upper_right = calc_center(first)
    upper_left = calc_center(second)
    lower_left = calc_center(third)
    lower_right = calc_center(forth)

    #代表点があればその点に沿って線を描画
    if upper_right is not None and upper_left is not None and lower_left is not None and lower_right is not None:
        result3 = cv2.line(input_image, (upper_right[0], upper_right[1]), (upper_left[0], upper_left[1]), (255,0,0), 5)
        result3 = cv2.line(result3, (upper_left[0], upper_left[1]), (lower_left[0], lower_left[1]), (255,0,0), 5)
        result3 = cv2.line(result3, (lower_left[0], lower_left[1]), (lower_right[0], lower_right[1]), (255,0,0), 5)
        result3 = cv2.line(result3, (lower_right[0], lower_right[1]), (upper_right[0], upper_right[1]), (255,0,0), 5)

        cv2.imshow("f",result3)
        cv2.imwrite("../save/f.jpg",result3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()