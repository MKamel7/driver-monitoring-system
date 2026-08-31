import cv2
#Opencv DNN
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

#Load Class Lists
classes= []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
#print("Objects list")
#print(classes)

#Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    #Get Frames
    ret, frame = cap.read()

    #Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    #filter the repeat detection        
    indexes = cv2.dnn.NMSBoxes(bboxes, scores,0.6, 0.4)  
    #print (indexes)
    #draw box with name around object
    for i in range(len(bboxes)):
        if i in indexes:
            x, y, w, h = bboxes[i]
            class_name = str(classes[class_ids[i]])
            if class_name =="cell phone":
                cv2.putText(frame, class_name, (x,y -10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

    #print("class ids", class_ids)
    #print("scores", scores)
    #print("Bboxes", bboxes)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
