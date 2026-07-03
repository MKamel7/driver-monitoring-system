#!/usr/bin/env python3
"""EAR node — subscribes to /driver_camera, publishes eye state on /ear_result.

Fixes vs. the original: the publisher is created once (it used to be
re-created inside the callback on every frame), and rospy.spin() is no
longer wrapped in a while loop.
"""

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from Include.ear import EarDetector

bridge = CvBridge()
detector = EarDetector()
pub = None


def on_frame(ros_frame):
    frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    _, eye_state = detector.process(frame)
    pub.publish(eye_state)


def main():
    global pub
    rospy.init_node("ear_node")
    pub = rospy.Publisher("ear_result", String, queue_size=1)
    rospy.Subscriber("driver_camera", Image, on_frame, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
