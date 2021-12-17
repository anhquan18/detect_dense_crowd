#!/usr/bin/env python

# ROS modules
import rospy
import os
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox, ObjectCount
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

# Standard library
import copy
import cv2
import time
import numpy as np
from pygame import mixer

# Realsesne
import pyrealsense2 as rs

def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        raise Exception('ERROR: no Depth camera found')

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    global pipeline, align, clipping_distance


class CrowdDetector(object):
    def __init__(self):
        self.pub = rospy.Publisher("crowd_information", String, queue_size=5)
        self.mitsu_pub = rospy.Publisher("mitsu_point", Point, queue_size=2)
        self.human_detector_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.human_detector_callback)
        #self.raw_img_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.img_receive_callback, queue_size = 1)
        self.raw_img_sub = rospy.Subscriber("image_raw", Image, self.img_receive_callback, queue_size = 1)
        self.image_raw = rospy.Publisher('image_raw', Image, queue_size=10)
        self.bridge = CvBridge()
        self.sailen = "/home/quan/catkin_ws/src/detect_dense_crowd/crowd_detection/script/yuriko_test/20210922_024553.mp3"
        self.mitsu = "/home/quan/catkin_ws/src/detect_dense_crowd/crowd_detection/script/yuriko_test/20210922_023949.mp3"
        mixer.init()
        mixer.music.load(self.sailen)

    def publish_image(self):
        while not rospy.is_shutdown():
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            global aligned_depth_frame

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            self.image_raw.publish(self.bridge.cv2_to_imgmsg(color_image, "bgr8"))

            #print type(color_frame)
            #print type(color_image)
            #print color_image.shape
            #print"depth img:", depth_image
            #print depth_image.shape
            #print depth_image[240][320]
    
    def img_receive_callback(self, img):
        self.img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

    def human_detector_callback(self, detection_datas):
        human_data = []

        for obj in detection_datas.bounding_boxes:
            if obj.Class == "person" and obj.probability >= 0.7:
                person_bounding_x = (obj.xmax - obj.xmin) # image pixel col position
                person_bounding_y = (obj.ymax - obj.ymin) # image pixel row position

                detected_human = {}
                #detected_human["position"] = (person_bounding_x/2, person_bounding_y/2)
                #detected_human["size"] = (person_bounding_x, person_bounding_y)
                detected_human["box"] = (obj.xmin, obj.xmax, obj.ymin, obj.ymax)

                human_data.append(copy.deepcopy(detected_human))

        self.crowd_data = []
        self.dense_crowd_detector(human_data)
        self.visualize_crowd()

    def dense_crowd_detector(self, human_data):
        overlap_thresh = 10

        while human_data:
            check_human = human_data.pop(0)
            for human in human_data:
                overlap_x = self.get_overlap(check_human["box"][:2], human["box"][:2])
                overlap_y = self.get_overlap(check_human["box"][2:], human["box"][2:])

                check_x = (check_human["box"][1] + check_human["box"][0])/2
                x_ = (human["box"][1] + human["box"][0])/2

                check_y = (check_human["box"][3] + check_human["box"][2])/2
                y_ = (human["box"][3] + human["box"][2])/2

                check_depth = aligned_depth_frame.get_distance(check_x, check_y)
                depth_ = aligned_depth_frame.get_distance(x_, y_)

                print "check human:", check_x, check_y, check_depth
                print "human:", x_, y_, depth_

                if (abs(check_x - x_) <= 350) and (abs(check_y - y_) <= 140) and (abs(check_depth - depth_) <= 0.7):
                    group = [copy.deepcopy(check_human), copy.deepcopy(human)]
                    if group not in self.crowd_data:
                        self.crowd_data.append(group)
                
                #if (overlap_x >= overlap_thresh and overlap_y >= overlap_thresh):
                #    group = [copy.deepcopy(check_human), copy.deepcopy(human)]
                #    if group not in self.crowd_data:
                #        self.crowd_data.append(group)

    def get_overlap(self, a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def visualize_crowd(self):
        window_name = "Camera"
        color = (0,0,255)
        thickness = 2

        draw_img = copy.deepcopy(self.img)

        for person1, person2 in self.crowd_data:
            min_x = min(person1["box"][0], person2["box"][0])
            min_y = min(person1["box"][2], person2["box"][2])

            max_x = max(person1["box"][1], person2["box"][1])
            max_y = max(person1["box"][3], person2["box"][3])

            #cv2.rectangle(draw_img, (person1["box"][0], person1["box"][2]), (person1["box"][1], person1["box"][3]), color, thickness)
            #cv2.rectangle(draw_img, (person2["box"][0], person2["box"][2]), (person2["box"][1], person2["box"][3]), color, thickness)
            cv2.rectangle(draw_img, (min_x, min_y), (max_x, max_y), color, thickness)

            point = Point()
            point.x = (min_x + max_x)/2
            point.y = (min_y + max_y)/2

            h1_d = aligned_depth_frame.get_distance((person1["box"][0]+person1["box"][1])/2, (person1["box"][2]+person1["box"][3])/2)
            h2_d = aligned_depth_frame.get_distance((person2["box"][0]+person2["box"][1])/2, (person2["box"][2]+person2["box"][3])/2)

            point.z = (h1_d + h2_d)/2

            print point.x, point.y, point.z
            self.mitsu_pub.publish(point)

        if self.crowd_data and not mixer.music.get_busy():
            mixer.music.play()
        cv2.imshow(window_name, draw_img)
        cv2.waitKey(1)

    
if __name__ == "__main__":
    rospy.init_node("SecurityCenter")
    setup_realsense()
    c_detector = CrowdDetector()
    c_detector.publish_image()
    pipeline.stop()
