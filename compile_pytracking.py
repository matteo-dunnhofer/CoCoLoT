"""
Written by Matteo Dunnhofer - 2021

Initialize CoCoLoT to compile PreciseRoiPooling modules
"""
from PIL import Image
import numpy as np
import vot
import vot_path
import sys
sys.path.append(vot_path.base_path)
from CoCoLoT_Tracker import CoCoLoT_Tracker, p_config

image = np.array(Image.open(vot_path.base_path + 'CoCoLoT/data/00000001.jpg'))

tracker = CoCoLoT_Tracker(image, vot.Rectangle(0, 0, 100, 100), p=p_config())
_ = tracker.tracking(image)

print('Done!')
