#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray

def callback(msg):
    points = msg.data
    points_2d = np.array([[points[0],points[1]], [points[2],points[3]], [points[4],points[5]], [points[6],points[7]]])
    #points_2d = np.array([[528.0,140.0], [328.0,140.0], [328.0,340.0], [528.0,340.0]])
    ret, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_mat,dist)
    br.sendTransform((tvec[0],tvec[1],tvec[2]), tf.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2]), rospy.Time.now(), "bebop2", "map")
    print tvec
    print rvec
    print "\n"

if __name__ == '__main__':
    rospy.init_node('calc_pose_node', anonymous=True)
    br = tf.TransformBroadcaster()

    points_3d = np.array([[2.25,2.25,0],[-2.25,2.25,0],[-2.25,-2.25,0],[2.25,-2.25,0]], dtype = np.float32)
    camera_mat = np.array([[547.883009, 0.000000, 430.632446], [0.000000, 539.585218, 249.283165], [0.000000, 0.000000, 1.000000]])
    dist = np.array([0.019232, 0.002345, 0.008232, 0.001804, 0.000000])

    rospy.Subscriber("marker_detect/points", Float32MultiArray, callback)
    rospy.spin()