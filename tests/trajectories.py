import json
import re
import pandas as pd
from collections import defaultdict, Counter
from constants import *
from colours import *

class GrandTour(object):

    '''Support for teams not there right now
    '''
    def __init__(self, grand_tour, base_path = './'):

        self.roster = {}
        self.roster_trajectory = {}
        self.stage_winners = []
        self.teams = defaultdict(list)
        self.grand_tour = grand_tour
        self.base_path = base_path

        
        temp, _ = self.get_single_day_results(stage_number=1)
        df = temp['Stage']
        for _, row in df.iterrows():
            self.roster_trajectory[row['name']]= {'stage':[], 'points':[], 'kom':[], 'gc': []}
            self.roster[row['name']] = {'score': 0, 'team': row['team']}
            self.teams[row['team']].append(row['name'])

        # print(pd.DataFrame(self.roster).T.sort_values(ascending=False, by='score'))
        self.final_score =  self.process_results()
        self.name_to_rank = {k: idx+1 for idx, k in enumerate(self.final_score.index.tolist())}
        self.final_race_results, _  =  self.get_single_day_results(stage_number=21)

        self.final_score['index'] = [i+1 for i in range(len(self.final_score))]

        temp = self.final_race_results['GC']
        self.final_score['gc_pos'] = [temp[temp['name'] == name]['rank'].iloc[0]
                                      if len(temp[temp['name'] == name]['rank']) else 10000
                                      for name in self.final_score.index]


    def _preprocess(self, df):

        '''Bad scraping logic but it is what it is
        '''
        data_dict = {'rank': [], 'name': [], 'team': []}
        for idx, row in df.iterrows():
            idx = int(idx)
            for j, v in enumerate(row):
                # is_name = re.match("r'\w+'", str(v).lower().strip())
                simple_name = "".join(str(v).split()).strip().replace("-", "").lower() # Marco Van-Basten
                simple_name = simple_name.replace("'", "") # Ben O'Connor
                if simple_name in ["dnf", "dns", "otl"]:
                    # THE DNS IS NOT GIVEN TO RANK
                    continue
                is_name = simple_name.isalpha()
                # print(v, is_name)
                if is_name  and str(v).split()[0].isupper():
                    data_dict['rank'].append(idx + 1)
                    data_dict['name'].append(v)
                    data_dict['team'].append(row[j+1])
                    break

            assert len(data_dict['rank']) == idx+1

        return pd.DataFrame(data_dict)

    def get_single_day_results(self, stage_number):

        '''
        '''
        grand_tour = self.grand_tour
        if grand_tour not in ["giro", "tdf", "vuelta"]:
            raise ValueError("Not one of the major Grand tours")

        keys = ["Stage", "GC", "Points", "Kom", "Youth"]
        results = {}
        fname = self.base_path + "{}-stage-{}.json".format(grand_tour, stage_number)
        with open(fname) as fp:
            d  = json.load(fp)

        for key in keys:
            results[key] =  self._preprocess(pd.DataFrame(d['results'][key]))

        return results, d['type']

    def process_results(self):
        
        start = 1
        end = 21
        for stage_number in range(start, end+1):
            data_dict, stage_type = self.get_single_day_results(stage_number)
            self.get_stage_scores(data_dict, 'ITT' in stage_type)
            print("stage {} winner: {}".format(stage_number, data_dict['Stage'].iloc[0]['name']))
        # # FINAL GC
        results = final_gc
        for idx, row in data_dict['GC'].iterrows():
            rider = row['name']
            if idx < len(results):
                self.roster[rider]['score'] += results[idx]


        # FINAL POINTS
        results = final_points
        for idx, row in data_dict['Points'].iterrows():
            rider = row['name']
            if idx < len(results):
                self.roster[rider]['score'] += results[idx]



        # FINAL MOUNTAINS
        results = final_kom
        for idx, row in data_dict['Kom'].iterrows():
            rider = row['name']
            if idx < len(results):
                self.roster[rider]['score'] += results[idx]

        # FINAL TEAMS
        # TODO
        answer = pd.DataFrame(self.roster).T.sort_values(ascending=False, by='score')
        return answer

    def get_stage_scores(self, data_dict, stage_type_tt):

        '''The leading 10 riders at each nominated intermediate sprint score
        fantasy points. This scoring category will reward both breakaway
        riders and also those riders going for a high placing in the Points
        jersey standings. Points are not awarded on time-trial stages.

        Breakaway points are awarded to riders who are in the lead
        break at 50% distance through the stage. A breakaway is
        defined as a group of 30 riders or less, with a measurable gap
        of five seconds to any groups behind them.

        '''
        self.stage_winners += data_dict['Stage'].iloc[:3]['name'].tolist()

        df = data_dict['GC']
        results = gc_results
        for idx, row in df.iterrows():
            rider = row['name']
            self.roster_trajectory[rider]['gc'].append(idx + 1)
            if idx < len(results):
                self.roster[rider]['score'] += results[idx]
        
        df = data_dict['Stage']
        results = stage_results
        for idx, row in df.iterrows():
            rider = row['name']
            self.roster_trajectory[rider]['stage'].append(idx + 1)
            if idx < len(results):
                self.roster[rider]['score'] += results[idx]


        df = data_dict['Points']
        results = point_results
        for idx, row in df.iterrows():
            rider = row['name']
            if idx < len(results):
                self.roster[rider]['score'] += results[idx]
                self.roster_trajectory[rider]['points'].append(idx + 1)
            else:
                self.roster_trajectory[rider]['points'].append(-1)

        df = data_dict['Kom']
        results = mountain_results
        for idx, row in df.iterrows():
            rider = row['name']
            if idx < len(results):
                self.roster[rider]['score'] += results[idx]
                self.roster_trajectory[rider]['kom'].append(idx+1)
            else:
                self.roster_trajectory[rider]['kom'].append(-1)


        # INTERMEDIATE SPRINTS: TODO
        # BREAKAWAY TODO
        # SUMMIT TODO
        if not stage_type_tt:
            intermediate_sprint_points = [20, 16, 12, 8, 6, 5, 4, 3, 2, 1]
            # The leading 10 riders at each nominated intermediate sprint score
            # fantasy points. This scoring category will reward both breakaway
            # riders and also those riders going for a high placing in the Points
            # jersey standings. Points are not awarded on time-trial stages.

            assist_points = [8, 4, 2]
            for i, v in enumerate(assist_points):
                # These points are awarded to riders who have a team rider finish in the
                # top 3 of the stage results. Riders must start the stage to score
                # points. These points are not given out for the Individual or Team Time
                # Trial Stages.
                team = data_dict['Stage'].iloc[i]['team']
                winner = data_dict['Stage'].iloc[i]['name']
                for rider in self.teams[team]:
                    # This is the guy that won -- you cannot assist yourself
                    if rider != winner:
                        self.roster[rider]['score'] += v

                # These points are awarded to riders who have a team rider in the top 3
                # of the Genral Classification at the end of each day. Riders must start
                # the stage to score points. These points are not given out for the
                # Individual Time Trial Stages.
                team = data_dict['GC'].iloc[i]['team']
                winner = data_dict['GC'].iloc[i]['name']
                for rider in self.teams[team]:
                    # This is the guy that won -- you cannot assist yourself
                    if rider != winner:
                        self.roster[rider]['score'] += v

                # TOP TEAMS : not included for now
                # These points are awarded to riders whose team is in the Top
                # 3 of the Teams Classification at the end of each day. Riders
                # must start the stage to score points. These points are not
                # given out for the Individual Time Trial Stages.
                # TODO



        # In 2014 this was changed so that there are three levels of stages,
        # each with its own point classification scheme. The first level,
        # presumably the flat stages, will award points to 20 riders on a
        # scale from 50 to 1 point. Level two stages will award points to the
        # top 15 riders on a scale of 25 to 1 and level three stages will
        # award points to the top 10 riders on a scale of 15 to 1
        # point. Points at intermediate sprints will follow a similar scale


    def get_points_for_my_team(self, team_lst, TOP):

        common = set(self.final_score.head(TOP).index).intersection(set(team_lst))
        print("How many in the top total score {}?".format(TOP), len(common))
        top_score = len(common)
        
        common = set(self.final_race_results['GC'].head(TOP).name).intersection(set(team_lst))
        print("How many in the GC {}?".format(TOP), len(common))
        gc_top = len(common)
        
        temp = {}
        mrr = []
        for name in team_lst:
            name = name.strip()
            if name not in self.roster:
                raise ValueError("BAD NAME: {}".format(name))
            temp[name] = self.roster[name]
            mrr.append(1/self.name_to_rank[name])

        df = pd.DataFrame(temp).T.sort_values(ascending=False, by='score')
        # print(df)
        print("Total score", df['score'].sum())
        print("mean reciprocal rank  MRR", sum(mrr))

        print(top_score, gc_top, df['score'].sum(), sum(mrr))
        return top_score, gc_top, df['score'].sum(), sum(mrr)
