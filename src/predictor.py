import os
import numpy as np
import tensorflow as tf
from utilities import load_model
from object_detection.utils import label_map_util

class Predictor:
    def __init__(self, 
                 path_to_saved_model = os.path.join('exported-models','v1','saved_model'),
                 path_to_labels = os.path.join('annotations','label_map.pbtxt')):
        
        self.path_to_saved_model = path_to_saved_model
        self.detect_fn = load_model(path_to_saved_model)
        self.path_to_labels=path_to_labels
        self.category_index = label_map_util.create_category_index_from_labelmap(path_to_labels,use_display_name=True)
        
    def get_detections(self, image_np):
        
        # Running the infernce on the image specified in the  image path
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        #print(detections['detection_classes'])
        return detections

    

