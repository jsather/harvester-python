""" train.py contains the Watchdog class, which is responsible for starting 
    the training and ensuring continuous operation. Running this script as main 
    starts the training process.
"""
import argparse 
import datetime
import os 
import signal 
import subprocess 
import sys 
import time
import warnings 

import psutil

def get_named_pids(name):
    """ Returns pids of running processes with given name. """
    named = []

    try: # Python 2.7.6
        running = psutil.get_pid_list() 
    except AttributeError: # Python 2.7.13
        running = psutil.pids() 

    for p in running: 
        try:
            if psutil.Process(p).name() == name: 
                named.append(p)
        except TypeError:
            if psutil.Process(p).name == name:
                named.append(p)
    
    return named

def kill_named_processes(name, keep=[], sig=signal.SIGKILL):
    """ Kills all processes with given name, except those with pids in keep
        list. 
    """
    try: # Python 2.7.6
        running = psutil.get_pid_list() 
    except AttributeError: # Python 2.7.13
        running = psutil.pids() 

    for p in running: 
        try:
            process = psutil.Process(p)
            if process.name() == name and p not in keep: 
                process.send_signal(sig)
        except: # Will get exception if process already terminated
            pass 

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """ Kills child processes of given pid. Code copied from:
        https://answers.ros.org/question/215600/how-can-i-run-roscore-from-python/
    """
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        print("Parent process does not exist.")
        return

    try:
        children = parent.get_children(recursive=True) # Python 2.7.6
    except AttributeError:
        children = parent.children(recursive=True) # Python 2.7.13

    for process in children:
        print("Killing child process: " + str(process))
        process.send_signal(sig)

class Watchdog(object):
    """ Class for managing processes during DDPG training process. """

    def __init__(self, headless=True, continue_training=False, 
        check_status_interval=10, reset_interval=86400):
        """ Initialize the watchdog. """
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully) 
        
        self.check_status_interval  = check_status_interval 
        self.continue_training = continue_training 
        self.headless = headless
        self.reset_interval = reset_interval # auto restart daily
        self.roscore_process = None  
        self.train_process = None 
        self.resets = 0
        
        progs = get_named_pids(name='python')
        self.preserve_pids = [p for p in progs if p < 1000 or p == os.getpid()]
    
    def start_roscore(self, delay=5):
        """ Starts roscore, which is needed to run the simulation or any
            python script using rospy (aka everything).
        """
        process = subprocess.Popen(['roscore'])
        time.sleep(delay)
        return process 

    def start_training(self, verbose=True):
        """ Starts training process. """
        if self.roscore_process or self.train_process: 
            print('Cannot start training; process already running.')
            sys.stdout.flush()
            return 

        if verbose:
            print('{} Starting training process...'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S')))
            sys.stdout.flush()
        
        self.roscore_process = self.start_roscore()

        cmd = cmd = ['python', '-m', 'ddpg.ddpg']
        if self.continue_training:
            cmd.append('--continue-training')
        if self.headless:
            cmd.append('--headless')

        self.train_process = subprocess.Popen(cmd)

        if verbose:
            print('Done.')
            sys.stdout.flush()
    
    def check_training(self, verbose=True):
        """ Checks to ensure training process and roscore are running. Returns
            True if both processes are running, False otherwise.
        """
        if verbose:
            print('{} Checking training process...'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S')))
            sys.stdout.flush()
        
        try: 
            train_running = self.train_process.poll() is None
        except AttributeError as e: 
            print('Error polling training process: {}'.format(e))
            train_running = False 
        
        try:
            core_running = self.roscore_process.poll() is None 
        except AttributeError as e:
            print('Error polling roscore process: {}'.format(e))
            core_running = False 
        
        if verbose:
            print('Done.')
            sys.stdout.flush()

        return train_running and core_running
    
    def stop_training(self, verbose=True, pid_genocide=True, delay=5):
        """ Kills training process at pid stored in memberdata, as well as 
            child processes. 
        """
        if verbose:
            print('{} Stopping training processes...'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S')))
            sys.stdout.flush()
        
        if self.roscore_process is not None: 
            try: 
                kill_child_processes(self.roscore_process.pid)
                self.roscore_process.terminate()
                self.roscore_process.wait() 
            except Exception as e:
                print('Error stopping roscore: {}'.format(e))
            self.roscore_process = None 
        else:
            warnings.warn('No roscore process in memberdata.', Warning)

        if self.train_process is not None: 
            try:
                kill_child_processes(self.train_process.pid)
                self.train_process.terminate()
                self.train_process.wait() 
            except Exception as e: 
                print('Error stopping training process: {}'.format(e))
            self.train_process = None
        else:
            warnings.warn('No training process in memberdata.', Warning)

        # Potentially redudant action to kill orpan processes
        if pid_genocide:
            if verbose: 
                print('Sending SIGKILL to any orphan processes...')
                sys.stdout.flush()

            kill_named_processes(name='python', keep=self.preserve_pids,
                sig=signal.SIGKILL)
            kill_named_processes(name='roslaunch', keep=self.preserve_pids,
                sig=signal.SIGKILL)
            kill_named_processes(name='move_group', keep=self.preserve_pids,
                sig=signal.SIGKILL)  
            kill_named_processes(name='robot_state_publisher', keep=self.preserve_pids,
                sig=signal.SIGKILL)       
            kill_named_processes(name='gzserver', keep=self.preserve_pids,
                sig=signal.SIGKILL) 
            kill_named_processes(name='gzclient', keep=self.preserve_pids,
                sig=signal.SIGKILL)   
            kill_named_processes(name='roscore', keep=self.preserve_pids,
                sig=signal.SIGKILL)

            if verbose:
                print('Done.')
                sys.stdout.flush()

        time.sleep(delay)
        if verbose:
            print('Done.')
    
    def restart_training(self, verbose=True):
        """ Stops and then starts training process. """
        self.stop_training(verbose=verbose) 
        self.start_training(verbose=verbose)
        
    def exit_gracefully(self, sig, frame):
        """ Kill spawned processes when program terminated. """
        print('{} Signal: {}. Killing training process and exiting'. format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'), 
            str(sig)))
        self.stop_training()
        sys.exit()
    
    def run(self):
        """ Starts and supervises training process. """
        print('{} Starting training process with options:'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S')))
        print('  continue_training: {}'.format(str(self.continue_training)))
        print('  headless: {}'.format(str(self.headless)))
        print('  check_status_interval: {}s'.format(
            str(self.check_status_interval)))
        print('  reset_interval: {0:.2f}hr'.format(
            self.reset_interval/60.0/60.0))
               
        start = time.time() 
        self.restart_training()
        self.continue_training = True 

        while True:
            try:
                if time.time() - start > self.reset_interval:
                    print('Reset interval: {0:.2f}hr reached.'.format(
                        self.reset_interval/60.0/60.0))
                    start = time.time() 
                    self.restart_training() 
                    self.resets += 1 
                    print('Resets: ' + str(self.resets)) 

                if not self.check_training(verbose=True):
                    self.restart_training()
                    self.resets += 1
                    print('Resets: ' + str(self.resets))
            except Exception as e: 
                print(e)
                self.restart_training()
                self.resets += 1
                print('Resets: ' + str(self.resets))
            
            time.sleep(self.check_status_interval)

def main():
    """ Runs watchdog. """
    parser = argparse.ArgumentParser(description='provide arguments for watchdog')
    parser.add_argument('--restart-training', 
        help='discard old training data and start fresh', 
        dest='continue_training',
        default=True, 
        action='store_false')
    parser.add_argument('--headless',
        help='do not show simulation while training', 
        default=False, 
        action='store_true')
    args = parser.parse_args() 
    
    watchdog = Watchdog(continue_training=args.continue_training, 
        headless=args.headless)
    watchdog.run()

if __name__ == '__main__':
    main()