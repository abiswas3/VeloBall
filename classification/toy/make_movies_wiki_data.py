import wikipedia as wp
import os
import pandas as pd
import numpy as np
import sys

def get_wiki_summary(query):
    try:
        temp = wp.summary(query)
        return temp
    except:
        print(query, "had issues")
        return None

def get_wiki_page(query):
    try:
        temp = wp.page(query)        
        return temp.content
    except:
        print(query, "had issues")
        return None
    
base_path = 'wiki_data/movies/'
files = [i for i in os.listdir(base_path) if i.split('.')[-1] == 'csv']

summaries = []
for fName in files:
    print(fName)
    df = pd.read_csv(base_path+fName)
    summaries += [get_wiki_page(i+ ' (film)') for i in df['Film'].values]
    print()

summaries = [i for i in summaries if i != None]
# np.save('wiki_data/movies/all_movies_big', summaries)
