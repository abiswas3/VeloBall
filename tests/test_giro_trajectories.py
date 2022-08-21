import sys
sys.path.append('../')

import pandas
import numpy as np

from trajectories import GrandTour
from colours import *

race = 'giro'
year = 2020
if race == 'giro':
    from giro_constants import *
else:
    from tour_constants import *

import pandas as pd
from collections import defaultdict, Counter
if __name__ == '__main__':

    fname = './processed_web_data/{}/'.format(year)
    gt = GrandTour(race, base_path = fname)
    df = pd.DataFrame(gt.roster_trajectory).T
    TOP = 20
    print(race)
    print('='*80)
    print(df)
