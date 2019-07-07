""" ROS/detector interface for getting rewards and/or showing feed.
    Author: Jonathon Sather
    Last Updated: 9/26/2018
"""
import collections
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

import agent.config as agent_cfg
import config as image_cfg
from detector.detector import Detector
from image_utils import Arrow

class FeedROS(object):
    def __init__(self, init_node=False, tf_session=None):
        """ Initialize detector, ros subscribers, and feed. """
        if init_node:
            rospy.init_node('video_ros_node')

        self.detector = Detector(session=tf_session)

        # Set up subscriptions
        self.fps = image_cfg.image_fps
        self.obs = collections.deque(maxlen=self.fps) # Holds 1 second of history
        self.step = None
        self.step_count = None
        self.new_step = False
        self.new_frame = False

        self.camera_sub = rospy.Subscriber('/harvester/camera1/image_raw',
            Image, self._process_image_data, queue_size=1)
        self.step_sub = rospy.Subscriber('/j2s6s200/step', String,
            self._process_step, queue_size=1)
        self.step_count_sub = rospy.Subscriber('/j2s6s200/step_count',
            String, self._process_step_count, queue_size=1)

        # Set up display
        self.reward = None
        self.reward_loc = None
        self.image_shape = image_cfg.image_shape
        self.reward_threshold = agent_cfg.reward_threshold
        self.detection_reward = agent_cfg.detection_reward

        self.line_type = image_cfg.line_type
        self.bb_colors = image_cfg.bb_colors
        self.bb_fps_const = image_cfg.bb_fps_const
        self.bb_fps = 0
        self.bb_time = time.time()

        # Reward animation
        self.r_animation_steps = image_cfg.reward_animation_steps
        self.r_animation_offset = image_cfg.reward_animation_offset
        self.r_animation_ptr = 0
        self.r_animation = False

        # Action display
        self.max_theta, self.max_phi = agent_cfg.hemi_action_bound
        self.arrow_ctr = image_cfg.arrow_center
        self.arrow_right_bg = Arrow(ctr=self.arrow_ctr, orient='right')
        self.arrow_up_bg = Arrow(ctr=self.arrow_ctr, orient='up')
        self.arrow_left_bg = Arrow(ctr=self.arrow_ctr, orient='left')
        self.arrow_down_bg = Arrow(ctr=self.arrow_ctr, orient='down')
        self.arrow_right = Arrow(ctr=self.arrow_ctr, orient='right', offset=3.0)
        self.arrow_up = Arrow(ctr=self.arrow_ctr, orient='up', offset=3.0)
        self.arrow_left = Arrow(ctr=self.arrow_ctr, orient='left', offset=3.0)
        self.arrow_down = Arrow(ctr=self.arrow_ctr, orient='down', offset=3.0)

        # Error display
        self.error_loc = image_cfg.error_loc
        self.error_radius = image_cfg.error_radius

        # Set up permanent window
        self.window = cv2.namedWindow('Harvester Vision', cv2.WINDOW_AUTOSIZE)

    def _process_image_data(self, ros_data):
        """ Callback function for camera topic. Receives raw input and outputs
            Numpy array of pixel intensities, including all images in queue.
        """
        flat = np.fromstring(ros_data.data, np.uint8)
        rgb = np.reshape(flat, (ros_data.height, ros_data.width, -1))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # Convert to opencv format
        self.obs.append(bgr)
        self.new_frame = True

    def _process_step(self, ros_data):
        """ Callback function for step topic. Receives string containing angle
            increments and stores them as tuple in memberdata.
        """
        self.step = eval(ros_data.data)

    def _process_step_count(self, ros_data):
        """ Callback function for step count topic. Receives string containing
            step count and stores as integer.
        """
        self.step_count = eval(ros_data.data)
        self.new_step = True

    def _update_fps(self):
        """ Updates fps stored in memberdata by taking new datapoint.
            Returns 1 for successful update, 0 otherwise.
        """
        cur = time.time()
        delta = cur - self.bb_time
        try:
            fps = 1 / delta
        except ZeroDivisionError: # Shouldn't happen but cautious when dividing
            print("Divide by zero! Skipping fps update.")
            return 0

        self.bb_time = cur
        self.bb_fps = (1 - self.bb_fps_const)*self.bb_fps + \
            self.bb_fps_const*fps
        return 1

    def _update_reward_animation(self):
        """ Updates reward animation parameters and returns scale and position
            to display in current frame.
        """
        r_vals = {}

        percent = float(self.r_animation_ptr)/self.r_animation_steps
        t = np.pi/4.0*percent - np.pi/6.0
        x = 10*np.sin(2*t)
        y = np.square(x)

        r_vals['loc'] = (int(self.reward_loc[0] + x), int(self.reward_loc[1] + y -
            self.r_animation_offset))
        r_vals['scale'] = 1.5
        r_vals['thickness'] = int(2*r_vals['scale'])
        r_vals['color'] = (0, 255, 0)

        self.r_animation_ptr += 1

        if self.r_animation_ptr >= self.r_animation_steps:
            self.r_animation = False
            self.r_animation_ptr = 0

        return r_vals

    def bb_to_reward(self, bbs):
        """ Converts list of bounding boxes to detection reward value. """
        reward = 0.0
        max_confidence = 0.0
        location = (-self.image_shape[0], -self.image_shape[1])
        for bb in bbs:
            if bb[0] and (bb[1] > max(self.reward_threshold, max_confidence)):
                max_confidence = bb[1]
                location = (bb[2][0], bb[2][1])
                reward = self.detection_reward
        return reward, location

    def draw_frame(self, im, bbs, action=None, display_action=True):
        """ Draws frame with latest bounding box predictions overlaid.
            Inputs:
                im = cv2 image
                bbs = strawberry detection result
                action = latest action
                display_action = flag indicating whether to display action
        """
        # Put bounding boxes around detections
        for res in bbs:
            bb = res[2]
            x = int(bb[0])
            y = int(bb[1])
            w = int(bb[2] / 2)
            h = int(bb[3] / 2)
            # thick = 1 + int(4.0 * (res[1] - 0.5)/(self.reward_threshold - 0.5))
            thick = 2
            try:
                cv2.rectangle(im, (x - w, y - h), (x + w, y + h),
                    self.bb_colors[res[0]], thick)
                cv2.rectangle(im, (x - w, y - h - 20),
                    (x - w + 40, y - h), self.bb_colors[res[0]], -1)
                cv2.putText(
                    im, '%.2f' % res[1],
                    (x - w + 3, y - h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, self.line_type)
            except OverflowError: # display red circle in bottom left
                cv2.circle(im, self.error_loc, self.error_radius, (0, 0, 255),
                    -1)
        cv2.putText(
            im, 'BBox FPS: %.2f' % self.fps,
            (self.image_shape[0] - 135, self.image_shape[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, self.line_type)

        # Update reward animation
        if self.reward is not None:
            if self.r_animation:
                r_vals = self._update_reward_animation()

                cv2.putText(
                    im, '+%d' % self.reward, r_vals['loc'],
                    cv2.FONT_HERSHEY_SIMPLEX, r_vals['scale'], r_vals['color'],
                    r_vals['thickness'], self.line_type)

        if display_action:
            if action is not None:
                theta, phi = action

                # Draw background arrows
                cv2.fillPoly(im, [self.arrow_right_bg.full_pts], (0, 0, 0))
                cv2.fillPoly(im, [self.arrow_up_bg.full_pts], (0, 0, 0))
                cv2.fillPoly(im, [self.arrow_left_bg.full_pts], (0, 0, 0))
                cv2.fillPoly(im, [self.arrow_down_bg.full_pts], (0, 0, 0))

                # Fill in area proportional to action
                percent_theta = abs(theta / self.max_theta)
                percent_phi = abs(phi / self.max_phi)

                if np.sign(theta) == 1:
                    cv2.fillPoly(
                        im, [self.arrow_right.get_pts(percent_theta)],
                        (255, 255, 255))
                else:
                    cv2.fillPoly(
                        im, [self.arrow_left.get_pts(percent_theta)],
                        (255, 255, 255))

                if np.sign(phi) == 1:
                    cv2.fillPoly(
                        im, [self.arrow_down.get_pts(percent_phi)],
                        (255, 255, 255))
                else:
                    cv2.fillPoly(
                        im, [self.arrow_up.get_pts(percent_phi)],
                        (255, 255, 255))
            else:
                cv2.putText(
                    im, 'RESET', (self.arrow_ctr[0] - 20, self.arrow_ctr[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, self.line_type)

        cv2.imshow('Harvester Vision', im)
        cv2.waitKey(10) # Tune this value

    def show_annotated_feed(self):
        """ Displays annotated feed coming from camera. """
        while len(self.obs) < self.fps:
            print("Waiting for images...")
            time.sleep(1)
        time.sleep(2) #TODO: Determine if still need this

        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False

                start = time.time()
                bbs = self.detector.detect(self.obs[-1]) # set to self.bb if doing parallel business

                if self.new_step:
                    self.new_step = False
                    self.reward, self.reward_loc = self.bb_to_reward(bbs)
                    self.r_animation = True
                    self.r_animation_ptr = 0

                latency = time.time() - start
                from_end = min(int(latency*self.fps) + 1, len(self.obs))

                self.draw_frame(
                    self.obs[-from_end],
                    bbs,
                    action=self.step,
                    display_action=True)

class RewardROS(object):
    def __init__(self, init_node=False, tf_session=None):
        if init_node:
            rospy.init_node('reward_ros_node')

        self.detector = Detector(session=tf_session)
        self.obs = None
        self.camera_sub = rospy.Subscriber('/harvester/camera1/image_raw',
            Image, self._process_image_data, queue_size=1)

    def _process_image_data(self, ros_data):
        """ Callback function for camera topic. """
        flat = np.fromstring(ros_data.data, np.uint8)
        rgb = np.reshape(flat, (ros_data.height, ros_data.width, -1))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # Convert to opencv format
        self.obs = bgr

    def bb_to_reward(self, bbs):
        """ Converts current list of bounding boxes to reward value. Currently,
            reward is implemented as the maximum confidence of a ripe
            strawberry.
        """
        max_confidence = 0.0
        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1]

        return max_confidence

    def get_reward(self, obs=None):
        """ Processes observation and returns detection component of reward.
            If no observation given, processes latest frame.
        """
        if obs is None:
            obs = self.obs
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

        try:
            bbs = self.detector.detect(obs)
        except Exception as e:
            print('Error retrieving bounding boxes')
            print(e)

        return self.bb_to_reward(bbs) # NOTE: Just returning reward without status info

def main():
    while not rospy.is_shutdown():
        time.sleep(0.1)

if __name__ == '__main__':
    main()
