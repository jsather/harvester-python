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
        
        self.net = build.TFNet(options, session=session)
    
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

class Evaluator(object):
    """ Class used to perform object detection tests and report results.
    """

    def __init__(self, detector, overlap_criterion=0.5, display=False):
        """ Initialize Evaluator. """
        self.detector = detector
        self.overlap_criterion = overlap_criterion
        self.display = display
        self.names = self.detector.meta.names
        self._generate_display_colors()

        # Testing stats
        self.confusion = np.zeros((2,2))
        self.dx_list = []
        self.dy_list = []
        self.dw_list = []
        self.dh_list = []
        self.iou_list = []

    def _generate_display_colors(self):
        """ Generates display colors for each class. """
        self.predict_colors = []
        self.gt_colors = []
        for i in range(self.detector.meta.classes):
            rand_binary = None
            while rand_binary is None or rand_binary == (0, 0, 0):
                rand_binary = (np.random.binomial(1, 0.5),
                    np.random.binomial(1, 0.5),
                    np.random.binomial(1, 0.5))
            self.predict_colors.append(
                (255*rand_binary[0], 255*rand_binary[1], 255*rand_binary[2]))
            self.gt_colors.append(
                (125*rand_binary[0], 125*rand_binary[1], 125*rand_binary[2]))

    def iou(self, box1, box2):
        """ Calculates IOU between two bounding boxes. """
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def draw_result(self, img, result, ground_truth):
        """ Overlays prediction and ground truth locations on test image. """
        for res in result:
            bb = res[2]
            x = int(bb[0])
            y = int(bb[1])
            w = int(bb[2] / 2)
            h = int(bb[3] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h),
                self.predict_colors[res[0]], 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                (x - w + 40, y - h), self.predict_colors[res[0]], -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, '%.2f' % res[1],
                (x - w + 3, y - h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

        for g in ground_truth:
            bb = g[1]
            x = int(bb[0])
            y = int(bb[1])
            w = int(bb[2] / 2)
            h = int(bb[3] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h),
                self.gt_colors[g[0]], 2)

    def update_statistics(self, predicted, ground_truth):
        """ Updates confusion matrix and error statistics using predicted and
            ground truth values for image.
        """
        # Predicted bbox classification
        for pd in predicted:
            tp = False # Assume false positive until proven otherwise
            for g in ground_truth:
                iou = self.iou(pd[2], g[1])
                if iou > self.overlap_criterion \
                    and pd[0] == g[0]:
                    tp = True
                    self.iou_list.append(iou)
                    self.dx_list.append(pd[2][0] - g[1][0])
                    self.dy_list.append(pd[2][1] - g[1][1])
                    self.dw_list.append(pd[2][2] - g[1][2])
                    self.dh_list.append(pd[2][3] - g[1][3])
            if tp:
                self.confusion[1, 1] += 1
            else:
                self.confusion[0, 1] += 1

        # Predicted gt classification
        for g in ground_truth:
            tp = False # Assume false negative until proven otherwise
            for pd in predicted:
                if self.iou(pd[2], g[1]) > self.overlap_criterion \
                    and pd[0] == g[0]:
                    tp = True
            if tp:
                continue # Already updated with predicted classification
            else:
                self.confusion[1, 0] += 1

    def get_labels(self, image_path, label_path):
        """ Returns array containing labels in text file at label_path. """
        image = cv2.imread(image_path)
        im_h, im_w, _ = image.shape

        with open(label_path, 'r') as f:
            labels_str = f.readlines()

        labels = []
        for line in labels_str:
            label_list = [float(i) for i in line.split()]
            class_no = int(label_list[0])
            x = label_list[1] * im_w
            y = label_list[2] * im_h
            w = label_list[3] * im_w
            h = label_list[4] * im_h
            labels.append((class_no, (x, y, w, h)))

        return labels

    def evaluate(self, image_path, label_path):
        """ Evaluates detector on given image. """
        ground_truth = self.get_labels(image_path, label_path)
        predicted = self.detector.detect(image_path)
        self.update_statistics(predicted, ground_truth)

        if self.display:
            image = cv2.imread(image_path)
            self.draw_result(image, predicted, ground_truth)
            cv2.imshow('Results Overlay: ' + image_path, image)
            cv2.waitKey(0)

    def evaluate_dataset(self, dataset, max_eval=None):
        """ Evaluates detector on given dataset.
            args:
                dataset = text file with each line containing location of test
                    images
                max_eval = maximum evaluations
            returns:
                statistics = dictionary containing confusion matrix and other
                    testing stats
        """
        print("Evaluating dataset using overlap threshold: " + str(self.overlap_criterion))
        with open(dataset, 'r') as f:
            image_paths = f.readlines()

        for idx, path in enumerate(image_paths):
            image_path = path.rstrip()
            if idx % 50 == 0:
                print(str(idx) + '/' + str(len(image_paths)))

            if max_eval is not None and idx > max_eval:
                return self.get_statistics()

            label_path = image_path[:-4] + '.txt'
            self.evaluate(image_path, label_path)
        return self.get_statistics()

    def get_statistics(self):
        """ Returns dictionary with confusion, precision, recall, and error
            histograms.
        """
        stats = {}
        stats['confusion'] = self.confusion
        stats['precision'] = self.confusion[1, 1] / (
            self.confusion[1, 1] + self.confusion[0, 1])
        stats['recall'] = self.confusion[1, 1] / (
            self.confusion[1, 1] + self.confusion[1, 0])
        stats['dx'] = self.dx_list
        stats['dy'] = self.dy_list
        stats['dw'] = self.dw_list
        stats['dh'] = self.dh_list
        stats['iou'] = self.iou_list
        return stats

    def display_histograms(self):
        """ Displays histograms for error statistics stored in memberdata. """
        import matplotlib.pyplot as plt

        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True) # Error stats: dx, dy, dw, dh
        ax1.hist(self.dx_list, bins='auto')
        ax1.set(xlabel='dx', ylabel='#')

        ax2.hist(self.dy_list, bins='auto')
        ax2.set(xlabel='dy')

        ax3.hist(self.dw_list, bins='auto')
        ax3.set(xlabel='dw')

        ax4.hist(self.dh_list, bins='auto')
        ax4.set(xlabel='dh')

        plt.figure(2) # IOU
        plt.hist(self.iou_list, bins='auto')
        plt.xlabel('iou')
        plt.ylabel('#')

        plt.show()

def main():
    """ Run pose prediction network on test images. """
    # NOTE: Haven't tested this since updated. May need some tweaking.
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-eval', default=10000, type=int)
    parser.add_argument('--hist', dest='hist', default=False, action='store_true')
    parser.add_argument('--display', dest='display', default=False, action='store_true')
    parser.add_argument('--overlap-thresh', default=0.5, type=float)
    parser.add_argument('--dataset', default='/mnt/storage/detector/dataset1', type=str)
    parser.add_argument('--save-dir', default='/mnt/storage/detector/results', type=str)
    args = parser.parse_args()

    detector = Detector()
    evaluator = Evaluator(detector, overlap_criterion=args.overlap_thresh,
        display=args.display)
    stats = evaluator.evaluate_dataset(args.test_path, max_eval=args.max_eval)
    print("Precision: " + str(stats['precision']))
    print("Recall: " + str(stats['recall']))

    print('Saving stats to ' + args.save_dir)
    with open(args.save_dir, 'w') as f:
        pickle.dump(stats, f)

    if args.hist:
        evaluator.display_histograms()

if __name__ == '__main__':
    main()
