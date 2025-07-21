import cv2
import os
import time

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# Define paths to model and label directories
default_path = 'models' # location of model weights and labels
model_name = 'helov1_efficientdet0_edgetpu.tflite'
label_name = 'ewasp_label.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.1
NUM_CLASSES = 3

# [FUNCTION] Main function
def main():
    
    # STEP 1: Load model and labels using pycoral.utils
    print('Loading {} with {} labels.'.format(model_path, label_path))
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    labels = read_label_file(label_path)
    inference_size = input_size(interpreter)

    # STEP 2: Open webcam
    cap = cv2.VideoCapture(0) # Default webcam has ID of 0

    # STEP 3: Loop through webcam camera stream and run model
    while cap.isOpened():
        time_cp1 = time.time()
        
        ret, frame = cap.read() # Read from webcam
        
        if frame is None:
            break # stop script if frame is empty
        else:
            
            # STEP 4: Preprocess image to the size and shape accepted by model
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, inference_size)

            # STEP 5: Let the model do the work
            time_cp2 = time.time()
            run_inference(interpreter, rgb_image.tobytes())
            time_cp3 = time.time()
            
            # STEP 6: Get objects detected from the model
            objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]

            # STEP 7: Label detected objects to frame
            image = append_objs_to_img(frame, inference_size, objs, labels)

            # STEP 8: Show labeled image to screen
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time_cp4 = time.time()

            # Calculate delays
            fps = round(1/(time_cp4 - time_cp1), 2)
            inference = round((time_cp3 - time_cp2) * 1000, 2)
            print(f"Speed = {fps} fps || Inference = {inference} ms")
    
    cap.release()
    cv2.destroyAllWindows()

# [FUNCTION] Modify image to label objs and score
def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.score > 0.6: # only draw item if confidence > 75%
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()