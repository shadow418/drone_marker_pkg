#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

if __name__ == '__main__':
    detector = cv2.AKAZE_create()
    
    temp_image = cv2.imread("../resource/en.jpg",0)
    edges = cv2.Canny(temp_image,100,200)

    temp_image2 = cv2.imread("../resource/1.png",0)
    edges2 = cv2.Canny(temp_image2,100,200)

    temp_image3 = cv2.imread("../resource/2.png",0)
    edges3 = cv2.Canny(temp_image3,100,200)

    temp_image4 = cv2.imread("../resource/3.png",0)
    edges4 = cv2.Canny(temp_image4,100,200)

    ret = cv2.matchShapes(edges, edges2, 1, 0)
    ret2 = cv2.matchShapes(edges, edges3, 1, 0)
    ret3 = cv2.matchShapes(edges, edges4, 1, 0)

    print ret
    print ret2
    print ret3

    cv2.imshow("1", edges)
    cv2.imshow("2", edges2)
    cv2.imshow("3", edges3)
    cv2.imshow("4", edges4)
    cv2.waitKey(0)

    #kp1 = detector.detect(temp_gray_image)
    #kp2 = detector.detect(temp_gray_image2)

    #result = cv2.drawKeypoints(temp_gray_image, kp1, None)
    #result2 = cv2.drawKeypoints(temp_gray_image2, kp2, None)

    #while(1):
    #    cv2.imshow("test",result)
    #    cv2.imshow("test2",result2)
    #    cv2.waitKey(1)
    
    

    """    
    cap = cv2.VideoCapture("../resource/video/color2.mp4")
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp1 = detector.detect(gray_image)
        result = cv2.drawKeypoints(gray_image, kp1, None)
        cv2.imshow("test",result)
        cv2.waitKey(1)
    """
    