#!/usr/bin/env python3
"""Result node — overlays the latest EAR and pose states on the camera
stream and displays the fused result window.
"""

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

bridge = CvBridge()
eye_state = "No State"
pose_state = "No State"


def on_ear(msg):
    global eye_state
    eye_state = msg.data


def on_pose(msg):
    global pose_state
    pose_state = msg.data


def on_frame(ros_frame):
    frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")

    alert = eye_state == "Drowsy" or pose_state == "Suspected Drowsiness"
    color = (0, 0, 255) if alert else (200, 50, 50)
    cv2.putText(frame, f"Eyes: {eye_state}", (10, 15),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color, 1)
    cv2.putText(frame, f"Pose: {pose_state}", (10, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color, 1)
    if alert:
        cv2.putText(frame, "ALERT", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Driver Camera", frame)
    cv2.waitKey(1)


def main():
    rospy.init_node("result_node")
    rospy.on_shutdown(cv2.destroyAllWindows)
    rospy.Subscriber("ear_result", String, on_ear, queue_size=1)
    rospy.Subscriber("pose_result", String, on_pose, queue_size=1)
    rospy.Subscriber("driver_camera", Image, on_frame, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
