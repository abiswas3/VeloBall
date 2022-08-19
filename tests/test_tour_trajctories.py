import sys
sys.path.append('../')

import pandas

from trajectories import GrandTour
from colours import *
from tour_constants import *
import pandas as pd
from collections import defaultdict, Counter
if __name__ == '__main__':

    YEAR = 2021
    TOP = 15
    for GT in ['tdf', 'vuelta', 'giro']:
        print("GT: {} YEAR: {}".format(GT, YEAR))
        gt = GrandTour(GT,
                       base_path = './processed_web_data/{}/'.format(YEAR))


        print(gt.final_score.head(TOP)[['score', 'gc_pos']])
        # df = pd.DataFrame(gt.roster_trajectory).T        
        # for rider in gt.final_score.head(TOP).index:
        #     print(rider)
        #     print("GC", df.loc[rider]['gc']) # GC and stage will always be correlated
        #     print("Stage", df.loc[rider]['stage'])            
        #     print("Kom", df.loc[rider]['kom'])
        #     print("Points", df.loc[rider]['points'])
        #     print()
        # print()

