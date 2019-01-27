""" evaluator.py contains methods to test both the ddpg policy and the static 
    detector. 

    Author: Jonathon Sather
    Last updated: 1/13/2019
"""
import argparse 
import csv
import datetime 
import os
import pickle 
import shutil 
import signal
import sys 
import time 

import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 

import config as test_cfg
import detector.config as detector_cfg
import detector.detector as detector 
import utils 

import pdb 

# TODO: 
# x  1) Debug detector tests
# x  2) Finish ddpg tests
#    2.5) Add analysis to ddpg tests: 
#         - fixation: positions to 3d coordinates 
#                     from there, can plot sequential positions on 3d plot (path) and visually determine domains of attraction?
#                                 or can run mean shift or some algorithm to find areas where tend to focus... look into this. dbscan or meanshift - read paper and watch vid
#                                 bottom line is need to find boundaries of domains of attraction, then classify pts as in regions or outside and calc avg rewards
#         - global exploration: positions to 3d coordinates
#                               what is the new idea? - just video, but keep coords just in case!
#    2.75) Make scripts to do clustering for fixation and global exploration
#    3) Debug ddpg tests
# x  4) Run detector tests
#    5) Run ddpg tests 
#       TODO: 
#       video for each baseline comparison
#    x  canopy density at mu=35
#    x  run fixation test
#       analysis scripts for fixation test
#       videos clips for global and local exploration
#       put results from each test in report!

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def detector_PR(output_dir, granularity=0.05, max_eval=None):
    """ Runs detector on test set, varying thresholds to create 
        precision-recall curve, and saves results.
    """
    dataset = test_cfg.image_test_set 

    detector_options = detector_cfg.df_options 
    detector_options['threshold'] = granularity
    dt = detector.Detector(options=detector_options) 

    thresholds = np.arange(granularity, 1.0, granularity).tolist()

    evaluator = DetectorEvaluator(dt, output_dir=output_dir)
    evaluator.evaluate_dataset(dataset, thresholds=thresholds, max_eval=max_eval)
    evaluator.save_results()
    
def detector_performance(output_dir):
    """ Runs detector performance test and saves the results. """
    dataset = test_cfg.image_test_set 
    dt = detector.Detector()

    evaluator = DetectorEvaluator(dt, output_dir=output_dir)
    evaluator.evaluate_dataset(dataset)
    evaluator.save_results()

def ddpg_baseline_compare(output_dir, weights_file, results_file):
    """ Runs ddpg baseline test and saves results. """
    import agent.agent_ros as agent_ros
    import agent.config as agent_cfg
    import ddpg.networks as networks
    import ddpg.config as ddpg_cfg
    import ddpg.noise as noise 
    import policy

    config = tf.ConfigProto(**ddpg_cfg.tf_cfg)
    config.gpu_options.allow_growth = True 

    with tf.Session(config=config) as session:
        np.random.seed(ddpg_cfg.np_seed)
        tf.set_random_seed(ddpg_cfg.tf_seed)

        [obs_shape, action_shape] = agent_cfg.hemi_state_shape
        action_bound = agent_cfg.hemi_action_bound
        OU_noise = noise.OrnsteinUhlenbeckActionNoise(
            mu=agent_cfg.mu,
            sigma=agent_cfg.sigma,
            theta=agent_cfg.theta)

        # initialize function approximators
        # embedding_network = networks.EmbeddingNetwork(session)
        embedding_network = None
        actor_network = networks.ActorNetwork(
            session,
            obs_shape,
            action_shape,
            action_bound,
            OU_noise,
            embedding=embedding_network)
        
        agent = agent_ros.HemiAgentROS(headless=True, feed=False, detector=True)
        lut_info = {'thetas': agent.lut_thetas, 'phis': agent.lut_phis, 
            'mask': agent.lut_mask}
        
        random1 = policy.Random1(action_bound=action_bound)
        random2 = policy.Random2(action_bound=action_bound, lut_info=lut_info)
        random3 = policy.Random3(action_bound=action_bound, lut_info=lut_info)
        custom1 = policy.Custom1(action_bound=action_bound, 
            detector=agent.detector_feedback.detector, lut_info=lut_info)
        custom2 = policy.Custom2(action_bound=action_bound,
            detector=agent.detector_feedback.detector, lut_info=lut_info)
        ddpg = policy.DDPG(session=session, actor=actor_network, 
            weights_file=weights_file)
        hybrid = policy.Hybrid(session=session, actor=actor_network, 
            weights_file=weights_file, action_bound=action_bound, 
            lut_info=lut_info)
        
        policies = [random1, random2, random3, custom1, custom2, ddpg, hybrid]
        evaluator = AgentEvaluator(policies=policies, 
            output_dir=output_dir, agent=agent, session=session,
            results_file=results_file, 
            episode_length=100, num_episodes=100, # 100, 100
            test_name='baseline comparison')
        evaluator.compare_policies()
        print('Comparison complete. Terminating program.')
        os.kill(os.getpid(), signal.SIGTERM)

def ddpg_canopy_compare(output_dir, weights_file, results_file):
    """ Runs ddpg on three different mean canopy densities to assess 
        how performance changes. 
        Run DDPG policy for 100 episodes on 3 canopy densities. Log episode
        lengths, as well as other performance stats, in case anything 
        interesting.
        NOTE: Requires premade plant models specified as model_mu#.rsdf
        BUG: Seems to be spawning first plant as previous model. 
        Temporary fix is to run 101 plants and discard first when analyzing
        data.
    """
    import agent.agent_ros as agent_ros
    import agent.config as agent_cfg
    import ddpg.networks as networks
    import ddpg.config as ddpg_cfg
    import ddpg.noise as noise 
    import policy

    config = tf.ConfigProto(**ddpg_cfg.tf_cfg)
    config.gpu_options.allow_growth = True 

    with tf.Session(config=config) as session:
        np.random.seed(ddpg_cfg.np_seed)
        tf.set_random_seed(ddpg_cfg.tf_seed)

        [obs_shape, action_shape] = agent_cfg.hemi_state_shape
        action_bound = agent_cfg.hemi_action_bound
        OU_noise = noise.OrnsteinUhlenbeckActionNoise(
            mu=agent_cfg.mu,
            sigma=agent_cfg.sigma,
            theta=agent_cfg.theta)

        # initialize function approximators
        # embedding_network = networks.EmbeddingNetwork(session)
        embedding_network = None
        actor_network = networks.ActorNetwork(
            session,
            obs_shape,
            action_shape,
            action_bound,
            OU_noise,
            embedding=embedding_network)
        
        agent = agent_ros.HemiAgentROS(headless=True, feed=False, detector=True)
        ddpg = policy.DDPG(session=session, actor=actor_network, 
            weights_file=weights_file)

        policies = [ddpg]
        plants = ['model_mu35'] #['model_mu5', 'model_mu20', 'model_mu50']

        for plant in plants:
            print('Evaluating on plant: ' + plant)
            test = 'canopy comparision ' + plant
    
            if plant == 'model_mu50': # add delay for loading big plant
                delay = 10
            else:
                delay = 0

            evaluator = AgentEvaluator(policies=policies, 
                output_dir=output_dir, agent=agent, session=session,
                results_file=results_file, episode_length=100, 
                num_episodes=101, plant_file=plant, test_name=test, 
                episode_delay=delay)
            evaluator.compare_policies()

        print('Comparison complete. Terminating program.')
        os.kill(os.getpid(), signal.SIGTERM)

def ddpg_fixation_analysis(output_dir, weights_file, results_file):
    """ Run ddpg policy on different plants and log special data for 
        analyzing fixation behavior. Save plant models!!
    """
    # NOTE: Let's reconsider this test... I think once per episode
    #       makes sense. Although I think that this may be harder to
    #       measure than I expected. How do I eliminate fixations
    #       from the starting location?? Maybe I can eliminate those?
    #       Or at least it will be something to talk about...
    #       Do I need to compare to random? What is the purpose in
    #       this case? maybe I do not. Eh might as well include it just
    #       in case...
    import agent.agent_ros as agent_ros
    import agent.config as agent_cfg
    import ddpg.networks as networks
    import ddpg.config as ddpg_cfg
    import ddpg.noise as noise 
    import policy

    config = tf.ConfigProto(**ddpg_cfg.tf_cfg)
    config.gpu_options.allow_growth = True 

    with tf.Session(config=config) as session:
        np.random.seed(ddpg_cfg.np_seed)
        tf.set_random_seed(ddpg_cfg.tf_seed)

        [obs_shape, action_shape] = agent_cfg.hemi_state_shape
        action_bound = agent_cfg.hemi_action_bound
        OU_noise = noise.OrnsteinUhlenbeckActionNoise(
            mu=agent_cfg.mu,
            sigma=agent_cfg.sigma,
            theta=agent_cfg.theta)

        # initialize function approximators
        # embedding_network = networks.EmbeddingNetwork(session)
        embedding_network = None
        actor_network = networks.ActorNetwork(
            session,
            obs_shape,
            action_shape,
            action_bound,
            OU_noise,
            embedding=embedding_network)
        
        agent = agent_ros.HemiAgentROS(headless=True, feed=False, 
            detector=True)
        agent.plant_interval = 10000 # make sure never auto-spawns plant
        
        random1 = policy.Random1(action_bound=action_bound)        
        ddpg = policy.DDPG(session=session, actor=actor_network, 
            weights_file=weights_file)

        policies = [ddpg] #[random1, ddpg]
        test = 'fixation analysis'
        
        for plant in range(500):
            evaluator = AgentEvaluator(policies=policies, 
                output_dir=output_dir, agent=agent, session=session,
                results_file=results_file, 
                episode_length=100, num_episodes=1,  #100 10
                test_name=test)
            
            # spawn new plant and copy to test directory
            evaluator.agent.plant.new()
            time.sleep(15)
            src = os.path.join(evaluator.agent.plant.model_dir, 'model.sdf')
            dst = os.path.join(evaluator.summary_dir, 
                'model' + str(plant) + '.sdf')
            shutil.copyfile(src, dst)

            evaluator.compare_policies(save_j=True, save_rewards=True, 
                random=False)

        print('Comparison complete. Terminating program.')
        os.kill(os.getpid(), signal.SIGTERM)

def ddpg_global_exploration(output_dir, weights_file, results_file):
    """ Run ddpg policy and log positions and actions at each timestep for
        analyzing global exploration patterns. Note: Also capture video to see
        if can find interesting patterns.
    """
    import agent.agent_ros as agent_ros
    import agent.config as agent_cfg
    import ddpg.networks as networks
    import ddpg.config as ddpg_cfg
    import ddpg.noise as noise 
    import policy
    
    config = tf.ConfigProto(**ddpg_cfg.tf_cfg)
    config.gpu_options.allow_growth = True 

    with tf.Session(config=config) as session:
        np.random.seed(ddpg_cfg.np_seed)
        tf.set_random_seed(ddpg_cfg.tf_seed)

        [obs_shape, action_shape] = agent_cfg.hemi_state_shape
        action_bound = agent_cfg.hemi_action_bound
        OU_noise = noise.OrnsteinUhlenbeckActionNoise(
            mu=agent_cfg.mu,
            sigma=agent_cfg.sigma,
            theta=agent_cfg.theta)

        # initialize function approximators
        # embedding_network = networks.EmbeddingNetwork(session)
        embedding_network = None
        actor_network = networks.ActorNetwork(
            session,
            obs_shape,
            action_shape,
            action_bound,
            OU_noise,
            embedding=embedding_network)
        
        agent = agent_ros.HemiAgentROS(headless=True, feed=False, 
            detector=True) 
        ddpg = policy.DDPG(session=session, actor=actor_network, 
            weights_file=weights_file)

        policies = [ddpg]
        test = 'global exploration analysis'
    
        evaluator = AgentEvaluator(policies=policies, 
            output_dir=output_dir, agent=agent, session=session,
            results_file=results_file, 
            episode_length=100, num_episodes=100,
            test_name=test)
        evaluator.compare_policies(save_j=True, save_a=True)
        print('Comparison complete. Terminating program.')
        os.kill(os.getpid(), signal.SIGTERM)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Eval Classes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DetectorEvaluator(object):
    """ Object for evaluating static detector. """
    
    def __init__(self, detector, output_dir, overlap_criterion=0.5, 
        display=False):
        """ Initialize detector evaluator. """
        self.detector = detector
        self.overlap_criterion = overlap_criterion
        self.display = display
        self.classes = self.detector.net.framework.meta['labels']

        self.output_dir = output_dir 
        self.summary_dir = os.path.join(self.output_dir, 
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.save_config()

        self._generate_display_colors()
        self.initialize_stats()

    def _generate_display_colors(self):
        """ Generates display colors for each class for visualization. """
        self.predict_colors = []
        self.gt_colors = []
        for i in range(len(self.classes)):
            rand_binary = None
            while rand_binary is None or rand_binary == (0, 0, 0):
                rand_binary = (np.random.binomial(1, 0.5),
                    np.random.binomial(1, 0.5),
                    np.random.binomial(1, 0.5))
            self.predict_colors.append(
                (255*rand_binary[0], 255*rand_binary[1], 255*rand_binary[2]))
            self.gt_colors.append(
                (125*rand_binary[0], 125*rand_binary[1], 125*rand_binary[2]))

    def display_histograms(self):
        """ Displays histograms for error statistics stored in memberdata. """
        for i in range(len(self.confusion)):
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True) # Error stats: dx, dy, dw, dh
            ax1.hist(self.dx_list[i], bins='auto')
            ax1.set(xlabel='dx', ylabel='#')

            ax2.hist(self.dy_list[i], bins='auto')
            ax2.set(xlabel='dy')

            ax3.hist(self.dw_list[i], bins='auto')
            ax3.set(xlabel='dw')

            ax4.hist(self.dh_list[i], bins='auto')
            ax4.set(xlabel='dh')

            plt.figure(2*(i + 1)) # IOU
            plt.hist(self.iou_list[i], bins='auto')
            plt.xlabel('iou')
            plt.ylabel('#')

            plt.show()
    
    def display_pr(self):
        """ Displays PR curve for confusion matrices stored in memberdata. """
        stats = self.get_statistics()
        p = np.array(stats['precisions'])
        r = np.array(stats['recalls'])

        plt.plot(r, p)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()     

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

    def evaluate(self, image_path, label_path, thresholds):
        """ Evaluates detector on given image. """
        ground_truth = self.get_labels(image_path, label_path)
        image = cv2.imread(image_path)
        predicted = self.detector.detect(image)
        self.update_statistics(predicted, ground_truth, thresholds) # update this to handle more classes
        
        if self.display:
            image = cv2.imread(image_path)
            self.draw_result(image, predicted, ground_truth) # update this!
            cv2.imshow('Results Overlay: ' + image_path, image)
            cv2.waitKey(0)

    def evaluate_dataset(self, dataset, thresholds=[0.5], max_eval=None):
        """ Evaluates detector on given dataset.
            args:
                dataset = text file with each line containing location of test
                    images
                max_eval = maximum evaluations
            returns:
                confusion matrix
        """
        print("Evaluating dataset using overlap threshold: " + 
            str(self.overlap_criterion) + " and confidence threshold(s): " +
            str(thresholds))
        
        self.initialize_stats(thresholds=thresholds)
        with open(dataset, 'r') as f:
            image_paths = f.readlines()

        for idx, path in enumerate(image_paths):
            image_path = path.rstrip()
            if idx % 50 == 0:
                print(str(idx) + '/' + str(len(image_paths)))

            if max_eval is not None and idx > max_eval:
                return self.get_statistics()

            label_path = image_path[:-4] + '.txt'
            self.evaluate(image_path, label_path, thresholds)
        return self.get_statistics()

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

    def get_statistics(self):
        """ Returns dictionary with confusion, precision, recall, and error
            histograms.
        """
        stats = {}
        stats['confusion'] = self.confusion
        stats['precisions'] = [x[1, 1] / (x[1, 1] + x[0, 1]) for x in 
            self.confusion]
        stats['recalls'] = [x[1, 1] / (x[1, 1] + x[1, 0]) for x in 
            self.confusion]
        stats['dx'] = self.dx_list
        stats['dy'] = self.dy_list
        stats['dw'] = self.dw_list
        stats['dh'] = self.dh_list
        stats['iou'] = self.iou_list
        stats['thresholds'] = self.thresholds
        return stats
    
    def initialize_stats(self, thresholds=[0.5]):
        """ Initializes memberdata for performance testing stats, allocating
            structures based on how many confidence thresholds will be
            evaluated. 
        """
        self.confusion = [np.zeros((2,2)) for x in range(len(thresholds))]
        self.dx_list = [[] for x in range(len(thresholds))]
        self.dy_list = [[] for x in range(len(thresholds))]
        self.dw_list = [[] for x in range(len(thresholds))]
        self.dh_list = [[] for x in range(len(thresholds))]
        self.iou_list = [[] for x in range(len(thresholds))]
        self.thresholds = thresholds

    def iou(self, box1, box2):
        """ Calculates IOU between two bounding boxes. """
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def save_config(self):
        """ Saves testing configuration. """
        with open(os.path.join(self.summary_dir, 'config.txt'), 'w') as f:
            f.write('overlap criterion: ' + str(self.overlap_criterion) + '\n')
            f.write('detector FLAGS: ' + str(self.detector.net.FLAGS) + '\n')

    def save_results(self):
        """ Saves results stored in memberdata. """
        results_file = os.path.join(self.summary_dir, 'detector_stats.pkl')
        print('Saving results to file ' + results_file)

        stats = self.get_statistics()
        with open(results_file, 'w') as f:
            pickle.dump(stats, f)

        pr = {'precision': stats.pop('precisions', None), 
            'recall': stats.pop('recalls', None),
            'thresholds': stats.pop('thresholds', None)}
        stats.pop('confusion') 
        
        for t, thresh in enumerate(self.thresholds):
            thresh_stats = {key:val[t] for (key, val) in stats.iteritems()}
            thresh_csv = os.path.join(self.summary_dir, 
                'detector_stats' + str(thresh) + '.csv')
            print('Saving results for threshold: ' + str(thresh) + ' to csv ' + 
                thresh_csv)
            utils.save_dict_as_csv(thresh_stats, thresh_csv)
        
        pr_csv = os.path.join(self.summary_dir, 'detector_pr.csv')
        print('Saving pr data to csv: ' + pr_csv)
        utils.save_dict_as_csv(pr, pr_csv)

    def update_statistics(self, predicted, ground_truth, thresholds):
        """ Updates confusion matrix and error statistics using predicted and
            ground truth values for image.
        """
        # Predicted bbox classification
        pdb.set_trace()
        for t, thresh in enumerate(thresholds):
            valid_pds = [pd for pd in predicted if pd[1] >= thresh]
            for pd in valid_pds:
                tp = False # Assume false positive until proven otherwise
                for g in ground_truth:
                    iou = self.iou(pd[2], g[1])
                    if iou > self.overlap_criterion \
                        and pd[0] == g[0]:
                        tp = True
                        self.iou_list[t].append(iou)
                        self.dx_list[t].append(pd[2][0] - g[1][0])
                        self.dy_list[t].append(pd[2][1] - g[1][1])
                        self.dw_list[t].append(pd[2][2] - g[1][2])
                        self.dh_list[t].append(pd[2][3] - g[1][3])
                if tp:
                    self.confusion[t][1, 1] += 1
                else:
                    self.confusion[t][0, 1] += 1

            # Predicted gt classification
            for g in ground_truth:
                tp = False # Assume false negative until proven otherwise
                for pd in valid_pds:
                    if self.iou(pd[2], g[1]) > self.overlap_criterion \
                        and pd[0] == g[0]:
                        tp = True
                if tp:
                    continue # Already updated with predicted classification
                else:
                    self.confusion[t][1, 0] += 1

class AgentEvaluator(object):
    """ Object for evaluating policies. """

    def __init__(self, policies, output_dir, agent, results_file=None, 
        session=None, episode_length=100, num_episodes=100, plant_file=None,
        test_name='test', episode_delay=0):
        """ Initialize evaluator object. 
            TODO: Add documentation for args!...
        """
        self.agent = agent
        self.policies = policies
        self.episode_delay = episode_delay
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.plant_file = plant_file
        self.session = session 
        self.results_file = results_file
        self.test_name = test_name
        self.test_results = {}

        self.output_dir = output_dir
        self.summary_dir = os.path.join(self.output_dir,
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if self.results_file:
            self.load_results(self.results_file)

        self.save_config()

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def compare_policies(self, save_rewards=False, save_o=False, 
        save_j=False, save_a=False, random=False, display=False):
        """ Runs policy evaluation on each policy in memberdata and saves
            results.
        """
        print('Comparing policies.')
        for policy in self.policies:
            if self.test_results.get(policy.name) is None:
                self.test_policy(policy, save_rewards=save_rewards, 
                    save_o=save_o, save_j=save_j, save_a=save_a,
                    random=random)
        print('Policy comparison complete.')
        
        if display:
            self.display_results(self.test_results, thresh=0)
            self.display_results(self.test_results, thresh=4)           

    def display_results(self, results, thresh=0):
        """ Summarizes results stored in memberdata. """
        # Rearrange data for plotting
        plot_data = {} 

        for policy, stats in results.iteritems():
            save = [x[0] for x in enumerate(stats['episode lengths']) 
                if x[1] > thresh]
            if thresh != 0:
                if plot_data.get('suicides') is None:
                    plot_data['suicides'] = {}
                plot_data['suicides'][policy] = \
                    len(stats['episode lengths']) - len(save)

            for metric, values in stats.iteritems():       
                pruned = [values[i] for i in save]         
                val = float(sum(pruned))/len(pruned)
                
                if plot_data.get(metric) is None:
                    plot_data[metric] = {}

                plot_data[metric][policy] = val
        
        fig_num = 0
        for metric, policies in plot_data.iteritems():
            plt.figure(fig_num)
            plt.bar(range(len(policies)), list(policies.values()), 
                align='center')
            plt.xticks(range(len(policies)), list(policies.keys()))
            if thresh != 0:
                plt.title(metric + ' (thresh = ' + str(thresh) + ')')
            else:
                plt.title(metric)
            fig_num += 1
        
        plt.show()

    def exit_gracefully(self, sig, frame):
        """ Saves results and closes tensorflow session before exiting. """
        print('Signal: ' + str(sig))
        self.save_results()
        try:
            self.session.close()
        except:
            pass 

        utils.kill_named_processes(name='roslaunch', sig=signal.SIGTERM)
        time.sleep(2)
        utils.kill_named_processes(name='roscore', sig=signal.SIGTERM)
        time.sleep(2)

        utils.kill_named_processes(name='roslaunch', sig=signal.SIGKILL)   
        utils.kill_named_processes(name='move_group', sig=signal.SIGKILL)  
        utils.kill_named_processes(name='robot_state_publisher', sig=signal.SIGKILL)       
        utils.kill_named_processes(name='gzserver', sig=signal.SIGKILL) 
        utils.kill_named_processes(name='gzclient', sig=signal.SIGKILL)   
        utils.kill_named_processes(name='roscore',  sig=signal.SIGKILL)

        sys.exit()

    def load_results(self, results_file):
        """ Loads results from pkl file to memberdata. """
        print('{} Loading results from: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            results_file))    
        with open(results_file, 'r') as f:
            self.test_results = pickle.load(f)
        print('Done.')
        sys.stdout.flush()

    def print_results(self, results):
        """ Summarizes results stored in memberdata. """
        for policy, stats in results.iteritems():
            print(str(policy))
            for metric, values in stats.iteritems():
                val = float(sum(values))/len(values)
                print(' ' + str(metric) + ': ' + str(val) + 
                    ' (averaged over ' + str(len(values)) + 
                    ' episodes)')

    def save_config(self):
        """ Saves testing configuration. """
        with open(os.path.join(self.summary_dir, 'config.txt'), 'w') as f:
            f.write('test name: ' + str(self.test_name) + '\n')
            f.write('episode length: ' + str(self.episode_length) + '\n')
            f.write('num episodes: ' + str(self.num_episodes) + '\n')
            f.write('results file: ' + str(self.results_file) + '\n')
            f.write('plant file: ' + str(self.plant_file) + '\n')
            for policy in self.policies:
                f.write('name: ' + policy.name + '\n')
                if policy.name == 'ddpg':
                    f.write('  weights file: ' + policy.weights_file + '\n')

    def save_csvs(self):
        """ Saves results stored in memberdata to csv. """
        policy_names = self.test_results.keys()

        for policy in policy_names:
            stats = self.test_results[policy]
            # remove = []

            # for key, val in stats.iteritems(): # what am i actually iterating through here...
            #     # TODO: Fix this and make it into useable form!! For at least r and j...
            #     if key == 'all_j': # could def do this more elegantly...
            #         filename = os.path.join(self.summary_dir, 
            #             policy + key + '.csv')
                    
            #         pdb.set_trace()
            #         positions = {'theta': [j[0] for j in val],
            #             'phi': [j[1] for j in val]}
            #         utils.save_dict_as_csv(positions, filename)
            #         # with open(filename, 'w+') as csvfile:
            #         #     writer = csv.writer(csvfile)
            #         #     for pos in val:
            #         #         writer.writerows(list(pos))
            #         remove.append(key) 

            #     elif key == 'all_rewards':
            #         filename = os.path.join(self.summary_dir, 
            #             policy + key + '.csv')
            #         pdb.set_trace()
            #         rewards = {'rewards': val}
            #         utils.save_dict_as_csv(rewards, filename)
            #         # with open(filename, 'w+') as csvfile:
            #         #     writer = csv.writer(csvfile)
            #         #     for reward in val:
            #         #         writer.writerows([reward])
            #         remove.append(key) 

            #     elif key[:3] == 'all':
            #         filename = os.path.join(self.summary_dir, 
            #             policy + key + '.csv')
            #         with open(filename, 'w+') as csvfile:
            #             writer = csv.writer(csvfile)
            #             writer.writerows(val)
            #         remove.append(key)
            #         #stats.pop(key)
            
            # for key in remove:
            #     stats.pop(key)

            filename = os.path.join(self.summary_dir, policy + '.csv')
            utils.save_dict_as_csv(stats, filename)

    def save_results(self):
        """ Saves results stored in memberdata to pkl file. """
        print('{} Saving results to: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.summary_dir))
        save_loc = os.path.join(self.summary_dir, 'test_results.pkl')
        with open(save_loc, 'w') as f:
            pickle.dump(self.test_results, f)
        self.save_csvs()
        print('Done.')
        sys.stdout.flush()

    def test_policy(self, policy, save_rewards=False, save_o=False, 
        save_j=False, save_a=False, random=False):
        """ Tests policy and saves data in summary dir. 
            Args:
                policy = Policy object to test
                save_rewards = boolean whether to save rewards for
                    each timestep
                save_o = boolean whether to save camera observations for
                    each timestep
                save_j = boolean whether to save hemi positions for each
                    timestep
                save_a = boolean whether to save actions for each timestep
           Returns None
        """
        print('Testing policy: ' + policy.name)
        ep_ave_rewards = []
        ep_length = []
        ep_max_rewards = []
        ep_steps_to_r = []
        ep_term_rewards = []
        ep_total_rewards = []
        
        # Only filled if specified by arg (to save space)
        all_rewards = [] 
        all_o = []
        all_j = []
        all_a = []

        for ep in range(self.num_episodes):
            print('Starting episode ' + str(ep))
            ep_rewards = []
            ep_o = []
            ep_j = []
            ep_a = []
            steps_to_r = -1
           
            o, j = self.agent.reset(random=random, plant_file=self.plant_file)

            if self.episode_delay:
                time.sleep(self.episode_delay)

            for step in range(self.episode_length):
                a = policy.get_action(o, j)
                [o, j], r, t, _ = self.agent.step(a)
                ep_rewards.append(r)
                ep_o.append(o)
                ep_j.append(j)
                ep_a.append(a)

                if steps_to_r < 0 and r == 1.0:
                    steps_to_r = step+1

                if t:
                    break

            print('Episode ' + str(ep) + ' complete. Updating statistics.')
            ep_ave_rewards.append(np.mean(ep_rewards))
            ep_length.append(step+1)
            ep_max_rewards.append(max(ep_rewards))
            ep_steps_to_r.append(steps_to_r)
            ep_term_rewards.append(ep_rewards[-1])
            ep_total_rewards.append(sum(ep_rewards))

            if save_rewards:
                all_rewards.append(ep_rewards)

            if save_o:
                all_o.append(ep_o)
            
            if save_j:
                all_j.append(ep_j)
            
            if save_a:
                all_a.append(ep_a)

        self.test_results[policy.name] = {
            'average rewards': ep_ave_rewards,
            'episode lengths': ep_length,
            'max rewards': ep_max_rewards,
            'steps to reward': ep_steps_to_r,
            'terminal rewards': ep_term_rewards, 
            'total rewards': ep_total_rewards}
        
        if save_rewards:
            self.test_results[policy.name]['all_rewards'] = all_rewards
        
        if save_o:
            self.test_results[policy.name]['all_obs'] = all_o
        
        if save_j:
            self.test_results[policy.name]['all_j'] = all_j
        
        if save_a:
            self.test_results[policy.name]['all_a'] = all_a 

        print('Test complete. Saving updated results.')
        self.save_results()

def main(args_dict):
    """ Runs tests specified by commandline args."""
    if args_dict['test'] == 'pr':
        print('Running detector precision recall test.')
        detector_PR(output_dir=args_dict['output_dir'])
    elif args_dict['test'] == 'stats':
        print('Running detector performance test.')
        detector_performance(output_dir=args_dict['output_dir'])
    elif args_dict['test'] == 'baseline':
        print('Running ddpg baseline comparison.')
        ddpg_baseline_compare(output_dir=args_dict['output_dir'], 
            weights_file=args_dict['weights_file'], 
            results_file=args_dict['results_file'])
    elif args_dict['test'] == 'canopy':
        print('Running ddpg canopy comparison.')
        ddpg_canopy_compare(output_dir=args_dict['output_dir'], 
            weights_file=args_dict['weights_file'], 
            results_file=args_dict['results_file'])
    elif args_dict['test'] == 'fixation':
        print('Running ddpg fixation analysis.')
        ddpg_fixation_analysis(output_dir=args_dict['output_dir'], 
            weights_file=args_dict['weights_file'], 
            results_file=args_dict['results_file'])
    elif args_dict['test'] == 'global' or args_dict['test'] == 'local':
        print('Running ddpg global/local exploration analysis.')
        ddpg_global_exploration(output_dir=args_dict['output_dir'], 
            weights_file=args_dict['weights_file'], 
            results_file=args_dict['results_file'])
    else:
        print('Invalid test specified.')
    
if __name__ == '__main__':
    # Parse command-line arguments and run tests
    parser = argparse.ArgumentParser(
        description='provide arguments for policy evaluation')
    parser.add_argument('--output-dir',
        help='directory for logging test info and results',
        default='/mnt/storage/testing')
    parser.add_argument('--results-file',
        help='file containing test results from current trial',
        default='',
        type=str)
    parser.add_argument('--test',
        help='specify which test to run. ' + 
            'Options: pr, stats, baseline, canopy, fixation, local',
        default='pr',
        type=str)
    parser.add_argument('--weights-file',
        help='file containing pretrained weights (leave empty to get latest)',
        default='',
        type=str)
    args_dict = vars(parser.parse_args())
    
    if args_dict['weights_file'] == '':
        args_dict['weights_file'] = utils.get_latest_weights(args_dict['output_dir'])
    main(args_dict)   

    # python evaluator.py --test 'baseline' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
    # python evaluator.py --test 'baseline' --results-file '/mnt/storage/testing/2019_01_11_05_26/test_results.pkl' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
    # python evaluator.py --test 'canopy' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
    # python evaluator.py --test 'fixation' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'