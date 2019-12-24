import rosbag
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
import tf

# Extra utility functions
from utility import *

def get_data(bag_filename, topic_name):

    msgs = []

    bag = rosbag.Bag(bag_filename)
    topics = set([topic_name ])

    for topic, msg, t in bag.read_messages(topics=topics):
            msgs.append(msg)

    bag.close()
    return msgs

class RobotLocalization(object):
    """
    Class used to interface with the rover. Gets sensor measurements through ROS subscribers,
    and transforms them into the 2D plane, and publishes velocity commands.
    """
    def __init__(self, world_map, bag_filename):
        """
        Initialize the class
        """
        # Internal variables
        self._T = tf.transformations.identity_matrix()
        self.Previous_T = tf.transformations.identity_matrix()

        self._marker_num = None
        self._T_tag2cam = None
        self._T_inter2world = None

        self._T_cam2bot   = np.array([[0,0,1,0],[-1,0,0,0.06],[0,-1,0,0],[0,0,0,1]])
        self._T_tag2inter = np.array([[1,0,0,0],[0, np.cos(0.281926), np.sin(0.281926), 0.0975868],
        [0, -np.sin(0.281926), np.cos(0.281926), -0.028265],[0,0,0,1]])

        # ROS publishers and subscribers
        self._world_map = world_map

        self.pose_msgs = get_data(bag_filename, "/tag_detections")
        self.bb_msgs = get_data(bag_filename, "/darknet_ros/bounding_boxes")

        self.poses = []
        self.pose_times =  []

        self.bbs = []
        self.pose_synced = []
        self.bb_times = []

        self.Previous_time = None

    def _tag_pose_callback(self):
        """
        Callback function for AprilTag measurements
        """
        for msg in self.pose_msgs:

            detections = msg.detections
            if (len(msg.detections)==0):
                continue

            exponential_coordinates = []
            translations = []
            for detection in detections:
                self._T_tag2cam = get_T(detection.pose.pose.pose)
                self._marker_num = detection.id
                current_header = detection.pose.header
                inter_pose = self._world_map[self._marker_num,:]
                inter_pose = np.squeeze(inter_pose)

                self._T_inter2world = get_inter2world(inter_pose)
                self._T = np.dot(np.dot(self._T_inter2world, self._T_tag2inter), np.linalg.inv(self._T_tag2cam))

                T = np.dot(tf.transformations.inverse_matrix(self.Previous_T), self._T)
                angle, direc, point = tf.transformations.rotation_from_matrix(T)
                translation = tf.transformations.translation_from_matrix(T)

                exponential_coordinate = direc*angle
                o = tf.transformations.translation_from_matrix(self._T)
                if o[2] < 0.697:
                    if self.Previous_time != None:
                        time_interval = detection.pose.header.stamp.to_sec() - self.Previous_time
                        angular_velocity = angle / time_interval
                        translational_velocity = np.linalg.norm(translation) / time_interval

                        if (np.abs(angular_velocity) < 0.9) and (translational_velocity < 3):
                            exponential_coordinates.append(exponential_coordinate)
                            translations.append(translation)
                    else:
                        exponential_coordinates.append(exponential_coordinate)
                        translations.append(translation)

            if len(exponential_coordinates):
                exponential_coordinates = np.array(exponential_coordinates)
                exponential_coordinates = np.mean(exponential_coordinates, axis=0)

                translations = np.array(translations)
                translations = np.mean(translations, axis=0)

                angle = np.linalg.norm(exponential_coordinates)
                direc = exponential_coordinate / angle

                T = tf.transformations.rotation_matrix(angle, direc)
                T[:3, 3] = translations
                self._T = np.dot(self.Previous_T, T)

                q = tf.transformations.quaternion_from_matrix(self._T)
                o = tf.transformations.translation_from_matrix(self._T)
                if q[0] < 0:
                    q = -q;

                self.poses.append(np.concatenate([q, o]))
                self.pose_times.append(msg.header.stamp.to_sec())

                self.Previous_T = self._T
                self.Previous_time = msg.header.stamp.to_sec()

        self.poses = np.array(self.poses)
        self.pose_times = np.array(self.pose_times)

    def bb_callback(self):
        for msg in self.bb_msgs:
            if len(msg.bounding_boxes) == 1 and msg.bounding_boxes[0].Class == 'chair':
                self.bbs.append([msg.bounding_boxes[0].xmin, msg.bounding_boxes[0].ymin,
                msg.bounding_boxes[0].xmax, msg.bounding_boxes[0].ymax, msg.bounding_boxes[0].probability])
                self.bb_times.append(msg.image_header.stamp.to_sec())

        self.bbs = np.array(self.bbs)
        self.bb_times = np.array(self.bb_times)

    def plot_results(self):
        print(len(self.bb_times))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.bb_times, self.pose_synced[: , 6], linewidth = 4)
        ax.grid(True)
        plt.show()

    def sync_data(self):
        data_len = self.bb_times.shape[0]
        for bb_index in range(data_len):
            bb_time = self.bb_times[bb_index]

            pose_index1 = np.where(self.pose_times > bb_time)[0][0]
            pose_index0 = pose_index1 - 1

            delta_t =  bb_time - self.pose_times[pose_index0]
            t = self.pose_times[pose_index1] -  self.pose_times[pose_index0]

            T0 = tf.transformations.quaternion_matrix(self.poses[pose_index0, 0:4])
            T0[:3, 3] = self.poses[pose_index0, 4:7]
            T1 = tf.transformations.quaternion_matrix(self.poses[pose_index1, 0:4])
            T1[:3, 3] = self.poses[pose_index1, 4:7]
            T01 = np.dot(tf.transformations.inverse_matrix(T0), T1)

            angle, direc, point = tf.transformations.rotation_from_matrix(T01)
            angle = angle*delta_t/t
            T = tf.transformations.rotation_matrix(angle, direc, point)
            T = np.dot(T0, T)

            q = tf.transformations.quaternion_from_matrix(T)
            o = tf.transformations.translation_from_matrix(T)
            if q[0] < 0:
                q = -q;
            self.pose_synced.append(np.concatenate([o, q]))

        self.pose_synced = np.array(self.pose_synced)
        np.savetxt('trajectory.txt', self.pose_synced, fmt ='%6.4f', delimiter=' ')
        np.savetxt('detection.txt', self.bbs, fmt =['%i', '%i', '%i', '%i', '%4.2f'], delimiter=' ')
        #np.savetxt('test.out', self.pose_synced)

def main(args):
    # Load parameters from yaml
    param_path = '/home/bear/catkin_ws/src/apriltag_localization/params/params.yaml'
    f = open(param_path,'r')
    params_raw = f.read()
    f.close()
    params = yaml.load(params_raw)
    world_map = np.array(params['world_map'])
    # Intialize the RobotControl object
    bag_filename ='/home/bear/catkin_ws/bags/SLAM2.bag'
    robotLocalization = RobotLocalization(world_map, bag_filename)
    robotLocalization._tag_pose_callback()
    robotLocalization.bb_callback()
    robotLocalization.sync_data()
    #robotLocalization.plot_results()
if __name__ == "__main__":
    main(sys.argv)
