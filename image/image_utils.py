""" display_utils.py contains utilities for the harvester camera display.

    Author: Jonathon Sather
    Last updated: 8/28/2018
"""
import numpy as np
import config as image_cfg 

def show_annotated_feed():
    """ Runs script to show annotated feed. Returns PID of process."""
    pwd = os.path.split(os.path.abspath(__file__))[0]
    loc = os.path.join(pwd, 'scripts', 'show_feed.py')
    process = subprocess.Popen(['nohup', loc],
        stdout=open('/dev/null', 'w'),
        stderr=open('logfile.log', 'a'),
        preexec_fn=os.setpgrp)
    return process.pid
    
def rotate(pt, theta):
    """ Rotates points theta radians. Looks like it rotates clockwise for
        raster coordinates.
        args:
             pt = 2x1 numpy array representing point
             theta = rotation angle in radians
        returns 2x1 numpy array of point after rotation
    """
    r_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    return np.matmul(r_matrix, pt)

class Arrow(object):
    """ Helper class to organize arrow drawings. """
    def __init__(self, ctr, orient, offset=0.0):
        """ Initialize the arrow. """
        self.ctr = ctr
        self.orient = orient
        self.offset = offset

        self.base_len = image_cfg.arrow_base_len
        self.base_width = image_cfg.arrow_base_width 
        self.tip_len = image_cfg.arrow_tip_len
        self.tip_width = image_cfg.arrow_tip_width
        self.base_offset = image_cfg.arrow_base_offset
        self.base_percent = self.base_len / (self.base_len + self.tip_len)

        # Reference points of arrow at 0 degrees from x about origin
        # For arrow pointing along positive x axis,
        #     ref_pt1 == top left corner of base
        #     ref_pt2 == bottom left corner of base
        #     ref_pt3 == top right corner of base
        #     ref_pt4 == bottom right corner of base
        #     ref_pt5 == top corner of tip
        #     ref_pt6 == bottom corner of tip
        #     ref_pt7 == tip of tip
        self.ref_pt1 = np.array([self.base_offset + self.offset,
            -self.base_width/2.0 + self.offset])
        self.ref_pt2 = np.array([self.base_offset + self.offset,
            self.base_width/2.0 - self.offset])
        self.ref_pt3 = np.array([self.base_offset + self.base_len + self.offset,
            -self.base_width/2.0 + self.offset])
        self.ref_pt4 = np.array([self.base_offset + self.base_len + self.offset,
            self.base_width/2.0 - self.offset])
        self.ref_pt5 = np.array([self.base_offset + self.base_len + self.offset,
            -self.tip_width/2.0 + (1 + np.sqrt(2))*self.offset])
        self.ref_pt6 = np.array([self.base_offset + self.base_len + self.offset,
            self.tip_width/2.0 - (1 + np.sqrt(2))*self.offset])
        self.ref_pt7 = np.array([self.base_offset + self.base_len +
            self.tip_len - np.sqrt(2)*self.offset, 0.0])

        self.full_pts = self.get_pts(1.0)

    def get_pts(self, percent):
        """ Returns points to "fill" the arrow up to specified percent
            of length.
        """
        if percent <= self.base_percent:
            pts = 4*[None]
            pts[0] = self.ref_pt1
            pts[1] = self.ref_pt1 + percent*np.array([self.base_len, 0])
            pts[2] = self.ref_pt2 + percent*np.array([self.base_len, 0])
            pts[3] = self.ref_pt2

        elif percent < 1.0:
            pts = 8*[None]
            pts[0] = self.ref_pt1
            pts[1] = self.ref_pt3
            pts[2] = self.ref_pt5
            pts[3] = self.ref_pt5 + \
                (percent - self.base_percent) * (self.ref_pt7 - self.ref_pt5)
            pts[4] = self.ref_pt6 + \
                (percent - self.base_percent) * (self.ref_pt7 - self.ref_pt6)
            pts[5] = self.ref_pt6
            pts[6] = self.ref_pt4
            pts[7] = self.ref_pt2

        else:
            pts = 7*[None]
            pts[0] = self.ref_pt1
            pts[1] = self.ref_pt3
            pts[2] = self.ref_pt5
            pts[3] = self.ref_pt7
            pts[4] = self.ref_pt6
            pts[5] = self.ref_pt4
            pts[6] = self.ref_pt2

        if self.orient == 'right':
            pts = [self.ctr + p for p in pts]
        elif self.orient == 'up':
            pts = [self.ctr + rotate(p, 3*np.pi/2) for p in pts]
        elif self.orient == 'left':
            pts = [self.ctr + rotate(p, np.pi) for p in pts]
        elif self.orient == 'down':
            pts = [self.ctr + rotate(p, np.pi/2) for p in pts]

        return np.array(pts, np.int32)
