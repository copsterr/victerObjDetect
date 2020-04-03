import cv2 as cv
import numpy as np
import time


image = cv.imread("traffic1.jpg")
width = image.shape[1]
height = image.shape[0]
scale = 0.00392

classes = None
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# net = cv.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
net = cv.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

outvid = cv.VideoWriter('detect.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

# read video
fpath = "./challenge_video.mp4"
cap = cv.VideoCapture(fpath)

while(cap.isOpened()):
    ret, image = cap.read()
    # image = cv.UMat(image)
    
    tstart = time.time()

    blob = cv.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
        
    outs = net.forward(get_output_layers(net))
    
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
    # apply non-max suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
    elapsed = time.time() - tstart
    fps = round(1/elapsed, 4)
    image = cv.putText(image, f"fps: {str(fps)}", (25, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)
    
    # display output image    
    outvid.write(image)
    cv.imshow("object detection", image)


    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break

cv.waitKey(1)
cap.release()
outvid.release()
cv.destroyAllWindows()





# # wait until any key is pressed
# cv.waitKey()
    
#  # save output image to disk
# cv.imwrite("object-detection.jpg", image)

# # release resources
# cv.destroyAllWindows()