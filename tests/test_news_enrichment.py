import json
import sys
sys.path.append('../')
# sys.path.append('/Users/aritrb/RelatedNews/src/Starman/src')
import os

import glob
import pprint as pp
import json
import logging
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

import crawler
import parsers
import parsers.utils as utils
# import starman

logging.basicConfig(level=logging.ERROR)

RED   = "\033[1;31m"
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"


# for fname in
# enricher = starman.NewsEnrichment()

##################### SIMULATE A CRAWLER USING A WEB BROWSER #####################
# url = "https://theintercept.com/2020/05/01/mdc-brooklyn-jail-coronavirus-medical-records/"
for stage_number in range(1,22):
    sys.stdout.write("\r")
    url = "https://www.procyclingstats.com/race/giro-d-italia/2022/stage-{}/result/result".format(stage_number)
    sys.stdout.write(url)    
    fname = '../saved_htmls/{}'.format(utils.get_hash(url))
    extractor = None
    c = crawler.HomePageHTMLRenderer(url, fname, override=False)

    filepath = sys.argv[1] if len(sys.argv) > 1 else fname 
    count = 0
    extractor = parsers.NewsExtractor()                
    with open(fname) as fp:
        temp = json.load(fp)            
        url = temp['url']
        page = temp['page']
        meta_data = {}
        nlp = None

    get_table_text = lambda x: [x for x in x.itertext()]
    extractor.process_page(url, page, schema_org_check=True, urls_crawled=[])
    df = pd.DataFrame([get_table_text(x) for x in extractor.tree.xpath("//table")[4].xpath("//tr")])
    df.to_json('Giro-stage-{}.json'.format(stage_number))
###################################################################################

