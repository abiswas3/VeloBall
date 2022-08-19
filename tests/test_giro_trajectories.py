import sys
sys.path.append('../')

import pandas
import numpy as np

from trajectories import GrandTour
from colours import *

race = 'tdf'
year = 2021
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

    # print("TOMS TEAM: won TT league")
    # team = toms_team
    # assert len(team) == 9
    # common = set(gt.final_score.head(TOP).index).intersection(set(team))
    # print("How many in the top {}?".format(TOP), len(common))
    # gt.get_points_for_my_team(team, TOP)
    # total_pods = 0
    # unique_pods = 0
    # for rider in toms_team:
    #     # print(rider)
    #     # print("Number of podiums", sum([1 if x <=3 else 0for x in df.loc[rider]['stage']]))
    #     total_pods += sum([1 if x <=3 else 0for x in df.loc[rider]['stage']])
    #     unique_pods += 1 if sum([1 if x <=3 else 0for x in df.loc[rider]['stage']]) > 0 else 0
    # print("Total Podiums: {} Unique Pods: {}".format(total_pods, unique_pods))
    # print('='*80)
    # print("MY TEAM: DID Median in small league")
    # team = my_team
    # assert len(team) == 9
    # gt.get_points_for_my_team(team, TOP)
    # total_pods = 0
    # unique_pods = 0
    # for rider in my_team:
    #     # print(rider)
    #     # print("Number of podiums", sum([1 if x <=3 else 0for x in df.loc[rider]['stage']]))
    #     total_pods += sum([1 if x <=3 else 0for x in df.loc[rider]['stage']])
    #     unique_pods += 1 if sum([1 if x <=3 else 0for x in df.loc[rider]['stage']]) > 0 else 0
    # print("Total Podiums: {} Unique Pods: {}".format(total_pods, unique_pods))
    # print()

    print('='*80)
    print(GREEN, "Best Teams in the world", RESET)
    print('='*80)
    score_df = gt.final_score.head(TOP)[['score', 'gc_pos']]
    mrr = []
    top = []
    gc = []
    score = []
    unique_pods_arr = []
    total_pods_arr = []
    for team in winning_teams:
        assert len(team) == 9
        top_score, gc_top, team_score, _mrr = gt.get_points_for_my_team(team, TOP)
        score.append(team_score)
        top.append(top_score)
        mrr.append(_mrr)
        gc.append(gc_top)
        total_pods = 0
        unique_pods = 0
        for rider in team:
            if rider in score_df.index:
                print(GREEN, rider, RESET)
            else:
                print(rider)
            # print("Number of podiums", sum([1 if x <=3 else 0for x in df.loc[rider]['stage']]))
            total_pods += sum([1 if x <=3 else 0for x in df.loc[rider]['stage']])
            unique_pods += 1 if sum([1 if x <=3 else 0for x in df.loc[rider]['stage']]) > 0 else 0
        total_pods_arr.append(total_pods)
        unique_pods_arr.append(unique_pods)
        print('='*80)
        print()

    print(score_df)
    # print("MRR", np.median(score), np.mean(score))
    # print("SCORE", np.median(top), np.mean(top))
    # print("GC", np.median(gc), np.mean(gc))
    # print("UNIQUe", np.median(unique_pods_arr), np.mean(unique_pods_arr))
    # print("Total", np.median(total_pods_arr), np.mean(total_pods_arr))
    # print('MOST POINTS WINNERS')
    # print(gt.final_score.head(30))
    # # print(gt.final_race_results['GC'].head(TOP))
    # # print()
    # print(gt.final_score.head(TOP).sort_values(by='gc_pos'))
    # print()
    # print("STAGE WINNER DISTRIBUTION")
    # print(pd.Series(Counter(gt.stage_winners)).sort_values(ascending=False))
