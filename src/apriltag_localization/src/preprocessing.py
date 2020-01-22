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
        self._T_tag2world = None

        self._T_cam2bot   = np.array([[0,0,1,0],[-1,0,0,0.06],[0,-1,0,0],[0,0,0,1]])

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

        self.plot_time = []
        self.plot_data1 = []
        self.plot_data2 = []

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
                inter_pose = self._world_map[self._marker_num, :]
                inter_pose = np.squeeze(inter_pose)

                self._T_tag2world = get_tag2world(inter_pose)
                self._T = np.dot(self._T_tag2world, np.linalg.inv(self._T_tag2cam))

                T = np.dot(tf.transformations.inverse_matrix(self.Previous_T), self._T)
                angle, direc, point = tf.transformations.rotation_from_matrix(T)
                translation = tf.transformations.translation_from_matrix(T)

                exponential_coordinate = direc*angle
                o = tf.transformations.translation_from_matrix(self._T)

                if o[2] < 0.697 and o[0] < -0.9 and o[0] > -4 and o[1] < -0.8 and o[1] > -4:
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
            for bounding_box in msg.bounding_boxes:
                if bounding_box.Class == 'laptop':
                    self.bbs.append([bounding_box.xmin, bounding_box.ymin,
                    bounding_box.xmax, bounding_box.ymax, bounding_box.probability])
                    self.bb_times.append(msg.image_header.stamp.to_sec())

        self.bbs = np.array(self.bbs)
        self.bb_times = np.array(self.bb_times)

    def plot_results(self):
        print(len(self.bb_times))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.pose_synced[:, 0], self.pose_synced[:, 1], 'bo' )#linewidth = 4
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
            T10 = np.dot(tf.transformations.inverse_matrix(T1), T0)

            if (delta_t/t > 0.5):
                angle, direc, point = tf.transformations.rotation_from_matrix(T10)
                angle = angle* (1 - delta_t/t)
                T = tf.transformations.rotation_matrix(angle, direc, point)
                T = np.dot(T1, T)
            else:
                angle, direc, point = tf.transformations.rotation_from_matrix(T01)
                angle = angle*delta_t/t
                T = tf.transformations.rotation_matrix(angle, direc, point)
                T = np.dot(T0, T)
            # angle, direc, point = tf.transformations.rotation_from_matrix(T01)
            # angle = angle*delta_t/t
            # T = tf.transformations.rotation_matrix(angle, direc)
            # T[:3, 3] = rotation_translation_vector(angle, direc, point)
            # T = np.dot(T0, T)

            q = tf.transformations.quaternion_from_matrix(T)
            o = tf.transformations.translation_from_matrix(T)
            if q[0] < 0:
                q = -q;
            self.pose_synced.append(np.concatenate([o, q]))

        self.pose_synced = np.array(self.pose_synced)
        #self.pose_synced[:,2] = 0.1
        np.savetxt('trajectories.txt', self.pose_synced[range(0,self.bb_times.size, 60)], fmt ='%6.4f', delimiter=' ')
        np.savetxt('detection.txt', self.bbs[range(0,self.bb_times.size, 60)], fmt =['%i', '%i', '%i', '%i', '%4.2f'], delimiter=' ')
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
    bag_filename ='/home/bear/catkin_ws/bags/ORBSLAM_data_set4.bag'
    robotLocalization = RobotLocalization(world_map, bag_filename)
    robotLocalization._tag_pose_callback()
    robotLocalization.bb_callback()
    robotLocalization.sync_data()
    robotLocalization.plot_results()
if __name__ == "__main__":
    main(sys.argv)
