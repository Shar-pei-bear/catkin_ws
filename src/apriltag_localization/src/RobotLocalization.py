#!/usr/bin/env python

import rospy
import roslib

import yaml
import numpy as np

import sys

from std_msgs.msg import (
    Header,
)

from apriltag_ros.msg import (
    AprilTagDetectionArray,
    AprilTagDetection,
)

from sensor_msgs.msg import Imu

from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    Pose,
    Twist,
)


# Extra utility functions
from utility import *

class RobotLocalization(object):
    """
    Class used to interface with the rover. Gets sensor measurements through ROS subscribers,
    and transforms them into the 2D plane, and publishes velocity commands.
    """
    def __init__(self, world_map):
        """
        Initialize the class
        """
        # Internal variables
        self._T = None

        self._marker_num = None
        self._T_tag2cam = None
        self._T_inter2world = None

        self._T_cam2bot   = np.array([[0,0,1,0],[-1,0,0,0.06],[0,-1,0,0],[0,0,0,1]])
        self._T_tag2inter = np.array([[1,0,0,0],[0, 0.9605, 0.2782, 0.0975868],[0, -0.2782, 0.9605, -0.028265],[0,0,0,1]])

        # ROS publishers and subscribers
        self._pub = rospy.Publisher("/robot_pose_corrected",PoseStamped,queue_size=10)
        self._world_map = world_map
        rospy.Subscriber("/tag_detections",AprilTagDetectionArray,self._tag_pose_callback)

        return

    def _tag_pose_callback(self,posearray):
        """
        Callback function for AprilTag measurements
        """
        if (len(posearray.detections)==0):
            return

        self._T_tag2cam = get_T(posearray.detections[0].pose.pose.pose)
        self._marker_num = posearray.detections[0].id
        current_header = posearray.detections[0].pose.header
        inter_pose = self._world_map[self._marker_num,:]
        inter_pose = np.squeeze(inter_pose)

        self._T_inter2world = get_inter2world(inter_pose)

        #self._T = np.dot(np.dot(self._T_inter2world, self._T_tag2inter), np.linalg.inv(np.dot(self._T_cam2bot, self._T_tag2cam)))
        self._T = np.dot(np.dot(self._T_inter2world, self._T_tag2inter), np.linalg.inv(self._T_tag2cam))

        StampedPose = make_pose_stamped_msg(self._T, current_header)
        if (self._T[2,3] < 1):
            self._pub.publish(StampedPose)
            # rospy.loginfo('cam 2 world is ')
            # rospy.loginfo(self._T)
            # rospy.loginfo(' ')

        # rospy.loginfo('The tag ID is')
        # rospy.loginfo(self._marker_num)
        # rospy.loginfo(' ')
        # rospy.loginfo('tag 2 world is ')
        # rospy.loginfo(np.dot(self._T_inter2world, self._T_tag2inter))
        # rospy.loginfo(' ')
        # rospy.loginfo('tag 2 cam is ')
        # rospy.loginfo(self._T_tag2cam)
        # rospy.loginfo(' ')
        # rospy.loginfo('cam 2 world is ')
        # rospy.loginfo(self._T)
        # rospy.loginfo(' ')

def main(args):
    rospy.init_node('apriltag_localization')

    # Load parameters from yaml
    param_path = rospy.get_param("~param_path")
    f = open(param_path,'r')
    params_raw = f.read()
    f.close()

    params = yaml.load(params_raw)
    world_map = np.array(params['world_map'])
    # Intialize the RobotControl object
    robotLocalization = RobotLocalization(world_map)
    rospy.spin()

if __name__ == "__main__":
    try:
        main(sys.argv)
    except rospy.ROSInterruptException: pass
