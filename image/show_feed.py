""" Script for running harvester feed. """

from image_ros import FeedROS 

feed = FeedROS(init_node=True)
feed.show_annotated_feed()
