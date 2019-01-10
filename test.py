""" test.py contains the Watchdog class, which is responsible for starting 
    the testing and ensuring continuous operation. Running this script as main 
    starts the testing process.
    
    Author: Jonathon Sather
    Last updated: 10/09/2018
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

# TO BE CONTINUED!