#!/usr/bin/env python3
"""Camera node — publishes webcam frames on /driver_camera at 10 Hz."""

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    rospy.init_node("camera_node")
    bridge = CvBridge()
    pub = rospy.Publisher("driver_camera", Image, queue_size=1)
    rate = rospy.Rate(10)

    cap = cv2.VideoCapture(rospy.get_param("~camera_index", 0))
    if not cap.isOpened():
        rospy.logfatal("Cannot open camera")
        return

    rospy.on_shutdown(cap.release)

    while not rospy.is_shutdown():
        ok, frame = cap.read()
        if not ok:
            rospy.logwarn_throttle(5, "No frame from camera")
            continue
        # Downscale to reduce transport and downstream inference cost.
        frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
        pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
        rate.sleep()


if __name__ == "__main__":
    main()
