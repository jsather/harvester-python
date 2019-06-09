""" detector.py contains classes for running and testing trained object
    detection network. Much of this module adapted from Darkflow:
    https://github.com/thtrieu/darkflow 
"""

import cv2 
import json 
import numpy as np 
import tensorflow as tf

import config as cfg 
from cython_utils.cy_yolo2_findboxes import box_constructor

class Detector(object):
    """ Detector class used to run pose prediction network feedforward on
        images. 
    """

    def __init__(self, session=None):
        """ Initialize detector. 
            Args: 
              session: tf session, or None if create new
        """
        self.cfg = {'pb': cfg.pb, 
                    'meta': cfg.meta,
                    'device': cfg.device,
                    'gpu_usage': cfg.gpu_usage}
        
        with open(self.cfg['meta'], 'r') as f:
            self.meta = json.load(f)

        if session is None:
            new_session = True
        else:
            self.session = Session 
            new_session = False 
        
        if new_session:
            self.graph = tf.Graph()
        device = self.cfg['device'] if self.cfg['gpu_usage'] > 0.0 else None 

        with tf.device(device):
            if new_session:
                with self.graph.as_default() as g:
                    self.inp, self.out = self._build(new_session=new_session)
                    self._config_tf()        
            else:
                self.inp, self.out = self._build(new_session=new_session)

    def _build(self, new_session=True):
        """ Build detector network from pb file (does not include
            post-processing)        
            Code partially copied from: 
            https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb       
        """
        with tf.gfile.GFile(self.cfg['pb'], 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
        
        tf.import_graph_def(graph_def, name='')
        
        inp = tf.get_default_graph().get_tensor_by_name('input:0')
        out = tf.get_default_graph().get_tensor_by_name('output:0')
        
        return inp, out 
    
    def _config_tf(self):
        """ Configure tensorflow for new session. """
        tf_config = {'allow_soft_placement': False,
                     'log_device_placement': False}
        
        if self.cfg['gpu_usage'] > 0:
            tf_config['gpu_options'] = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.cfg['gpu_usage'])
            tf_config['allow_soft_placement'] = True 
        else:
            tf_config['device_count'] = {'GPU': 0}
        
        cfg = tf.ConfigProto(**tf_config)
        cfg.gpu_options.allow_growth = True 
        self.session = tf.Session(config=cfg) 
        self.session.run(tf.global_variables_initializer())
    
    def _find_boxes(self, net_output):
        """ Find boxes from network output. Calls cython utility for 
            fast post processing.
        """
        meta = self.meta 
        boxes = list() 
        boxes = box_constructor(meta, net_output)
        return boxes 
    
    def _process_box(self, b, h, w, threshold):
        """ Process bounding box candidate, and output in my format. """
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = self.meta['labels'][max_indx]
        if max_prob > threshold:
            # trim out-of-bounds
            left  = int ((b.x - b.w/2.) * w)
            right = int ((b.x + b.w/2.) * w)
            top   = int ((b.y - b.h/2.) * h)
            bot   = int ((b.y + b.h/2.) * h)
            if left  < 0    :  left = 0
            if right > w - 1: right = w - 1
            if top   < 0    :   top = 0
            if bot   > h - 1:   bot = h - 1
            
            # output stuff in my format
            w = right - left 
            h = bot - top 
            x = top + int(round(w/2.))
            y = left + int(round(h/2.))
            return (max_indx, max_prob, (y, x, h, w))
        return None

    def _resize_input(self, image):
        """ Format image for network. """
        h, w, c = self.meta['inp_size']
        imsz = cv2.resize(image, (w, h))
        imsz = imsz / 255.
        imsz = imsz[:,:,::-1]
        return imsz 
    
    def detect(self, image, threshold=None):
        """ Runs pose prediction network on image and returns processed output
            in useful format. 
        """
        if threshold is None:
            threshold = self.meta['thresh']

        # Set up image for network
        h, w, _ = image.shape 
        image = self._resize_input(image)
        net_input = np.expand_dims(image, 0)
        feed_dict = {self.inp: net_input} 

        # Run network and post-processing
        out = self.session.run(self.out, feed_dict)[0]
        boxes = self._find_boxes(out) 
        boxes_info = []
        for box in boxes:
            processed = self._process_box(box, h, w, threshold)
            if processed is None:
                continue
            boxes_info.append(processed)
        return boxes_info 