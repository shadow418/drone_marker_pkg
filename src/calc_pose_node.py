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
    ret, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_mat,dist)
    br.sendTransform((tvec[0],tvec[1],tvec[2]), tf.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2]), rospy.Time.now(), "bebop2", "marker1")
    print tvec
    print rvec
    print "\n"

if __name__ == '__main__':
    rospy.init_node('calc_pose_node', anonymous=True)
    br = tf.TransformBroadcaster()

    points_3d = np.array([[0,0,0],[2,0,0],[0,0,2],[2,0,2]], dtype = np.float32)
    camera_mat = np.array([[537.292878, 0.000000, 427.331854], [0.000000, 527.000348, 240.226888], [0.000000, 0.000000, 1.000000]])
    dist = np.array([0.004974, -0.000130, -0.001212, 0.002192, 0.000000])

    rospy.Subscriber("marker_detect/points", Float32MultiArray, callback)
    rospy.spin()