#!/usr/bin/env python3
"""Head-pose node — subscribes to /driver_camera, publishes attention state
on /pose_result. Processes every 4th frame to halve inference load
(matching the original design intent).
"""

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from Include.head_pose import HeadPoseEstimator

FRAME_SKIP = 4

bridge = CvBridge()
estimator = HeadPoseEstimator()
pub = None
_counter = 0


def on_frame(ros_frame):
    global _counter
    _counter += 1
    if _counter < FRAME_SKIP:
        return
    _counter = 0

    frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    _, state, _direction = estimator.process(frame, draw=False)
    pub.publish(state)


def main():
    global pub
    rospy.init_node("pose_estimation_node")
    pub = rospy.Publisher("pose_result", String, queue_size=1)
    rospy.Subscriber("driver_camera", Image, on_frame, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
