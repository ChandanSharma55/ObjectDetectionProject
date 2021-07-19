#Import the required libraries for Object detection infernece
import os
import cv2
from predictor import Predictor
from object_detection.utils import visualization_utils as viz_utils

def return_output_for_video_frames(predictor_model, image_np, MIN_CONF_THRESH):    
    detections = predictor_model.get_detections(image_np)
    #print(detections['detection_classes'])
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          predictor_model.category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=MIN_CONF_THRESH,
          agnostic_mode=False,
          line_thickness=2)
    return image_np_with_detections

predictor_model = Predictor(path_to_saved_model = os.path.join('exported-models','v1','saved_model'))
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Our Fruit Extractor', return_output_for_video_frames(predictor_model, frame, 0.6))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
print('Done')
cv2.destroyAllWindows() 