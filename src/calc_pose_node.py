#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
import tf
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray

def callback(msg):
    points = msg.data
    points_2d = np.array([[points[0],points[1]], [points[2],points[3]], [points[4],points[5]], [points[6],points[7]], [points[8],points[9]], [points[10],points[11]]])
    #points_2d = np.array([[1060.0,340.0], [660.0,340.0], [660.0,740.0], [1060.0,740.0]])
    ret, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_mat,dist)

    R = cv2.Rodrigues(rvec)[0]
    projMat = np.array([[R[0][0], R[0][1], R[0][2], 0.0], [R[1][0], R[1][1], R[1][2], 0.0], [R[2][0], R[2][1], R[2][2], 1.0]] , dtype = np.float32)
    eulerAngles = cv2.decomposeProjectionMatrix(projMat)[6]
    print eulerAngles
    br.sendTransform((-tvec[0],-tvec[1],tvec[2]), tf.transformations.quaternion_from_euler(math.radians(-eulerAngles[0]), 0, math.radians(eulerAngles[2])), rospy.Time.now(), "bebop2", "map")
    
    #print tvec
    #print rvec
    print "\n"

if __name__ == '__main__':
    rospy.init_node('calc_pose_node', anonymous=True)
    br = tf.TransformBroadcaster()

    #points_3d = np.array([[2.1,0,2.1],[-2.025,0,2.025],[-2.025,0,-2.025],[2.1,0,-2.1],[2.1,0,-2.1],[2.1,0,-2.1]], dtype = np.float32)
    points_3d = np.array([[0.45,0,0.90], [-0.45,0,0.90], [-0.45,0,-0.90], [0.45,0,-0.90], [-0.45,0,0], [0.45,0,0]], dtype = np.float32)
    camera_mat = np.array([[1215.7, 0, 952.3442],[0, 1227.4, 542.1231],[0.0, 0.0, 1.0]])
    dist = np.array([0.0535, -0.1581, 0.3051, 0, 0])

    rospy.Subscriber("marker_detect/points", Float32MultiArray, callback)
    rospy.spin()