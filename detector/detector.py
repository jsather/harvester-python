""" detector.py contains classes for running and testing trained object
    detection network.

    Author: Jonathon Sather
    Last updated: 9/26/2018
"""
import argparse
import os 

import cv2 
import darkflow.net.build as build
# import net.build as build 
import numpy as np 
import pickle 

import config as cfg 

class Detector(object):
    """ Detector class used to run pose prediction network feedforward on
        images. 
    """

    def __init__(self, options=None, session=None):
        """ Initialize detector. 
            Args: 
              session: tf session, or None if create new
              cfg: config options, if None, use defaults
        """
        if options is None:
            options = cfg.df_options
        
        # if session is None:
        #     session = tf.Session()
        # self.session = session 
        
        # pdb.set_trace()
        # self.inp, self.out = self._build(
        #     pb_file=cfg.pb, 
        #     input_tensor='input:0',
        #     output_tensor='output:0'
        # )        
        self.net = build.TFNet(options, session=session)

        import pdb 
        pdb.set_trace()

    def _build(self, pb_file, input_tensor, output_tensor):
        """ Build detector network from pb file.        
            Code partially copied from: 
            https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb       
        
            Args:
                model_name = name of directory with pretrained model
                input_tensor = name of input tensor of pretrained model
                output_tensor = name of output tensor of pretrained model
            Returns:
                input tensor, output tensor, trainable variables
        """
        with tf.gfile.GFile(pb_file, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
            
        inp = tf.get_default_graph().get_tensor_by_name(in_tensor)
        out = tf.get_default_graph().get_tensor_by_name(out_tensor)

        return inp, out
    
    def _convert_output(self, orig):
        """ Converts darkflow prediction output into useful format. 
            orig: 
              [{'topleft':{'y':int, 'x':int}, 
                'bottomright':{'y':int, 'x':int}, 
                'label':str, 
                'confidence':float}, ...]
            return:
              [(class, confidence, (y, x, h, w)), ...], 
                where class,x,y,w,h=int, confidence=float
        """
        out = []
        for bb in orig:
            w = bb['bottomright']['x'] - bb['topleft']['x']
            h = bb['bottomright']['y'] - bb['topleft']['y']
            x = bb['topleft']['y'] + int(round(w/2.0))
            y = bb['topleft']['x'] + int(round(h/2.0))
            conf = bb['confidence']
            class_ = int(bb['label'] == 'ripe')

            out.append((class_, conf, (y, x, h, w)))

        return out 

    def detect(self, image):
        """ Runs pose prediction network on image and returns processed output
            in useful format. 
        """
        df_out = self.net.return_predict(image) 
        return self._convert_output(df_out)
