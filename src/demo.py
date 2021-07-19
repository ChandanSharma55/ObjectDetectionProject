import os
import cv2
from predictor import Predictor
from utilities import load_image_into_numpy_array
from object_detection.utils import visualization_utils as viz_utils

def show_detections(predictor, image_folder, MIN_CONF_THRESH = 0.6):
    for image_path in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_path)
        image_np = load_image_into_numpy_array(image_path)
        detections = predictor.get_detections(image_np)           
        
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              predictor.category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=MIN_CONF_THRESH,
              agnostic_mode=False,
              line_thickness=2)
        #plt.figure()
        #plt.imshow(image_np_with_detections)
        width = 840
        height = 840
        dsize = (width, height)
        original = cv2.resize(image_np, dsize, interpolation = cv2.INTER_AREA)
        output = cv2.resize(image_np_with_detections, dsize, interpolation = cv2.INTER_AREA)
        cv2.imshow("image", cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        cv2.imshow("detected objects",cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        #plt.show()
    print('Done')
    cv2.destroyAllWindows()

model1 = Predictor(path_to_saved_model = os.path.join('exported-models','v1','saved_model'))
model3 = Predictor(path_to_saved_model = os.path.join('exported-models','v3','saved_model'))
show_detections(predictor = model1, image_folder = os.path.join('test_images'))


