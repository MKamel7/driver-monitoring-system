'''
Drowsiness Detection with computer vision using dlib 
'''
# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================
from argparse import ArgumentParser
from math import dist
from cv2 import (COLOR_BGR2GRAY, FONT_HERSHEY_COMPLEX, cvtColor, VideoCapture, resize, INTER_CUBIC,
                destroyAllWindows, imshow, polylines, putText, waitKey)
from dlib import get_frontal_face_detector, shape_predictor
from numpy import array
import time

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================
'''
N_Initial_Frames: Number of Frames to Calculate initial EAR (20 ... 40)
Holding_Frames: Number of Frames to wait when detecting Drowsiness ((20 ... 40) depends on FPS)  
Drowsiness_Factor: % of Initial EAR to be Compared with the Current EAR to detect Drowsiness (0.5 ... 0.8)
Ignored_Initial_EAR_Values: Any Value of EAR Less then That number will be Ignored (0.2 ... 0.1)
'''
N_Initial_Frames = 30
Holding_Frames = 20
Drowsiness_Factor = 0.79
Ignored_Initial_EAR_Values = 0.2

# ==============================================================================
# -- Static Variables ----------------------------------------------------------
# ==============================================================================
class static:
    INITIAL_EAR = 0
    INITIAL_COUNTER = 0
    D_COUNTER = 0
    SUM_EAR = 0
    SC=0
    AC=0
    STATE = "NO STATE"


# ==============================================================================
# -- Dlib Face Detector --------------------------------------------------------
# ==============================================================================
dlib_face = shape_predictor("shape_predictor_68_face_landmarks.dat")
face_detector = get_frontal_face_detector()

# ==============================================================================
# -- Camera Setup --------------------------------------------------------------
# ==============================================================================
# ap = ArgumentParser()
# ap.add_argument("-w", "--webcam", type=int, default=0)
# args = vars(ap.parse_args())
# cap = VideoStream(src=args["webcam"]).start()
cap = VideoCapture(0) # opencv
# cap = VideoStream(usePiCamera=True).start()  # For Raspberry Pi

def Get_Eye(face_landmark_part):
    right_eye = array([(face_landmark_part(36).x, face_landmark_part(36).y),
                        (face_landmark_part(37).x, face_landmark_part(37).y),
                        (face_landmark_part(38).x, face_landmark_part(38).y),
                        (face_landmark_part(39).x, face_landmark_part(39).y),
                        (face_landmark_part(40).x, face_landmark_part(40).y),
                        (face_landmark_part(41).x, face_landmark_part(41).y)])
    left_eye = array([(face_landmark_part(42).x, face_landmark_part(42).y),
                        (face_landmark_part(43).x, face_landmark_part(43).y),
                        (face_landmark_part(44).x, face_landmark_part(44).y),
                        (face_landmark_part(45).x, face_landmark_part(45).y),
                        (face_landmark_part(46).x, face_landmark_part(46).y),
                        (face_landmark_part(47).x, face_landmark_part(47).y)])

    return right_eye, left_eye

def EAR_Calc(C_Right, C_Left):
    A = dist((C_Left[1]), (C_Left[5]))
    B = dist((C_Left[2]), (C_Left[4]))
    C = dist((C_Left[0]), (C_Left[3]))
    Left_EAR = (A+B)/(2.0*C)
    A = dist((C_Right[1]), (C_Right[5]))
    B = dist((C_Right[2]), (C_Right[4]))
    C = dist((C_Right[0]), (C_Right[3]))
    Right_EAR = (A+B)/(2.0*C)
    return (Left_EAR + Right_EAR)/2.0

def F_Initial_EAR(C_avg_EAR):
    """
    Name:   F_Initial_EAR_Calc
    Args:   Current Aye Aspect Ratio
    Job:    Calculates the The Initial EAR from the current EAR Without the Blinking Frames (Run only at the Beginning)
    Return: None
    """
    # When the Counter is reached the Threshold
    if static.INITIAL_COUNTER == N_Initial_Frames:
        # Calculate the Average Initial EAR from the Sum of the Current EAR
        static.INITIAL_EAR = round((static.SUM_EAR/N_Initial_Frames), 3)
        # Clear the Sum
        static.SUM_EAR = 0
        # Reset the Counter
        static.INITIAL_COUNTER = 0

    # When the Counter is Less Than the Threshold with Ignoring the Blinking Frames
    elif (static.INITIAL_COUNTER < N_Initial_Frames) and (C_avg_EAR > Ignored_Initial_EAR_Values):
        # Add the Current EAR Value to the Sum
        static.SUM_EAR += C_avg_EAR
        # Increment the Counter
        static.INITIAL_COUNTER += 1



def F_Get_State(C_Current_EAR, C_INITIAL_EAR):
    """
    Name:   F_Get_State
    Args:   Current Eye Aspect Ratio & Initial Eye Aspect Ratio
    Job:    Compere The Current EAR with Initial EAR o Get the State of the Driver
    Return: None
    """
    # Initial State Suppose No Driver
    static.STATE = "No Driver"

    # When the Eye is About to be Closed
    if C_Current_EAR < (C_INITIAL_EAR*Drowsiness_Factor):
        # Maybe the Driver is Blinking But Still Awake
        static.STATE = "Blinking"
        # Increment the Drowsiness Counter
        static.D_COUNTER += 1
        # When the Drowsiness Counter Reaches the Threshold (Not Blinking)
        if static.D_COUNTER > Holding_Frames:
            # The Driver is Drowsy
            static.STATE = "Drowsy"

    # When the Eye is Opened
    elif C_Current_EAR >= (C_INITIAL_EAR*Drowsiness_Factor):
        # Reset the Drowsiness Counter
        static.D_COUNTER = 0
        # The Driver is Awake
        static.STATE = "Awake"


def F_Get_Face(frame):
    gray = cvtColor(frame, COLOR_BGR2GRAY)
    faces = face_detector(gray)
        
    if faces:
        # Get landmarks on one face
        face_landmark = dlib_face(gray, faces[0])
        
        # Eyes index 
        right_eye, left_eye = Get_Eye(face_landmark.part)        
        
        # Calculate the EAR
        avg_EAR =EAR_Calc(right_eye,left_eye) 
        
        # at the start Calculate the Initial EAR
        if static.INITIAL_EAR == 0:
            F_Initial_EAR(avg_EAR)
        F_Get_State(avg_EAR, static.INITIAL_EAR)
        
        polylines(frame, [right_eye], 1, (0, 255, 0), 1)
        polylines(frame, [left_eye], 1, (0, 255, 0), 1)

        # Intial EAR
        putText(frame, 'initial_EAR='+(str(static.INITIAL_EAR)), (10, 20),
            FONT_HERSHEY_COMPLEX, 0.7, (100, 0, 100), 2)
        # Current EAR
        putText(frame, ("EAR="+(str(round(avg_EAR,3)))), (10, 40),
            FONT_HERSHEY_COMPLEX, 0.6, (100, 0, 100), 2)
        # FPS
        putText(frame, (str(static.STATE)), (10, 60),
            FONT_HERSHEY_COMPLEX, 0.6, (100, 0, 100), 2)
        
    return frame


# ==============================================================================
# -- Main Loop -----------------------------------------------------------------
# ==============================================================================
while True:
    ret, frame = cap.read()
    start = time.time()
    frame = F_Get_Face(frame)
    end = (time.time() - start)
    putText(frame, 'FPS='+(str(round(1/end,3))), (10, 80),
            FONT_HERSHEY_COMPLEX, 0.6, (100, 0, 100), 2)
    imshow('Result', frame)
    
    if waitKey(1) == ord('q'):
        destroyAllWindows()
        cap.release()
        break
