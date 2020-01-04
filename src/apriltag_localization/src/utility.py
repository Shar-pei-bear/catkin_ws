#!/usr/bin/env python
"""
Utility functions for manipulating poses in ROS.
Used mostly for extending coordinate transformations beyond the scope of transformations.py.
Written by Alex Zhu (alexzhu(at)seas.upenn.edu)
"""

import numpy as np

import roslib
import rospy
from std_msgs.msg import (
    Header,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Quaternion,
)
from tf.transformations import *

def get_t_R(pose):
    """
    Returns the translation vector (4x1) and rotation matrix (4x4) from a pose message
    """
    t=np.transpose(np.matrix([pose.position.x,pose.position.y,pose.position.z,0]))
    quat=[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    R_full=quaternion_matrix(quat)
    R=R_full
    return t,R

def get_T(pose):
    """
    Returns the transformation matrix (4x4) from a pose message
    """
    t=np.array([pose.position.x,pose.position.y,pose.position.z])
    #rospy.loginfo('translation is ')
    #rospy.loginfo(t)
    quat=[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    T = quaternion_matrix(quat)
    T[0:3, -1] = t
    return T

def get_tag2world(pose):
    theta = pose[2] * np.pi / 180.

    T_tag2inter = np.array([[1,0,0,0],[0, np.cos(pose[3]), np.sin(pose[3]), np.cos(pose[3])*0.1016],
    [0, -np.sin(pose[3]), np.cos(pose[3]), -np.sin(pose[3])*0.1016],[0,0,0,1]])

    T_inter2world = np.array([[np.sin(theta), 0, -np.cos(theta), pose[0]],
    [-np.cos(theta), 0, -np.sin(theta), pose[1]], [0, 1, 0, 0], [0, 0, 0, 1]])

    T = np.dot(T_inter2world, T_tag2inter)
    return T

def make_pose_stamped_msg(T,current_header):
    """
    Returns a pose stamped message from a translation vector and rotation matrix (4x4) for publishing.
    NOTE: Does not set the target frame.
    """
    pose_stamped_msg=PoseStamped()
    pose_stamped_msg.header=current_header
    pose_msg=Pose()

    t = np.squeeze(T[0:3, -1])
    pose_msg.position.x=t[0]
    pose_msg.position.y=t[1]
    pose_msg.position.z=t[2]

    quat=quaternion_from_matrix(T)
    pose_msg.orientation.x=quat[0]
    pose_msg.orientation.y=quat[1]
    pose_msg.orientation.z=quat[2]
    pose_msg.orientation.w=quat[3]
    pose_stamped_msg.pose=pose_msg

    return pose_stamped_msg

def rotation_translation_vector(angle, direc, point):
    """
    Return translational vector to rotate about axis defined by point and direction.
    """
    w = np.asarray([[0, -direc[2], direc[1]],[direc[2], 0, - direc[0]],[-direc[1], direc[0], 0]])
    v = np.cross(point[0:3], direc)
    o = np.dot((np.eye(3)*angle + (1 - np.cos(angle))*w + (angle - np.sin(angle))*np.dot(w, w)), v)

    return o
