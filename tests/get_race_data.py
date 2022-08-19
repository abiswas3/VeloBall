import json
import sys
sys.path.append('../PeterParker/')
import os

import glob
import pprint as pp
import json
import logging
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from pathlib import Path

import crawler
import parsers
import parsers.utils as utils
# import starman

logging.basicConfig(level=logging.ERROR)

GT = 0 # GIRO
# GT = 1 # TOUR
# GT = 2 # VUELTA
YEAR = 2020

WEB_LINK = [
    [
        'processed_web_data/{}/giro',
        "https://www.procyclingstats.com/race/giro-d-italia/{}/stage-{}/result/"
    ],    
    [
        'processed_web_data/{}/tdf',
        "https://www.procyclingstats.com/race/tour-de-france/{}/stage-{}"
    ],
    [
        'processed_web_data/{}/vuelta',
        "https://www.procyclingstats.com/race/vuelta-a-espana/{}/stage-{}"
    ]
    
]

##################### SIMULATE A CRAWLER USING A WEB BROWSER #####################

c = crawler.BaseHTMLRenderer()
for stage_number in range(1, 22):
    ######################################3
    base_name, url = WEB_LINK[GT]
    url = url.format(YEAR, stage_number)
    base_name = base_name.format(YEAR)
    print(url)
    fname = '../saved_htmls/{}'.format(utils.get_hash(url))
    extractor = None
    c.get_main_page(url, fname, override=False)
    ######################################
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else fname
    count = 0
    extractor = parsers.DocumentExtractor()

    try:
        with open(fname) as fp:
            temp = json.load(fp)
            url = temp['url']
            page = temp['page']
            meta_data = {}
            nlp = None
    except:
        continue

    get_table_text = lambda x: [x  for x in x.itertext()]
    extractor.process_page(url, page, schema_org_check=True, urls_crawled=[])
    print()
    
    print([x for x in extractor.tree.xpath("//div[@class='sub']")[0].itertext()])
    stage_type = [x for x in extractor.tree.xpath("//div[@class='sub']")[0].itertext()][2]
    
    
    print("\nSTAGE TYPE", stage_type)
    save_json = {'type': stage_type, 'results': {}}
        
    tables = extractor.tree.xpath("//table[@class='results basic moblist10']/tbody")
    print("Found {} tables ".format(len(tables)))
    d = {0: "Stage", 1: "GC", 2: "Points", 3: "Kom", 4: "Youth"}
    for index in d:
        df = pd.DataFrame([get_table_text(x) for x in tables[index].xpath("tr")])
        save_json['results'][d[index]] = df.to_dict()


    try:
        pakainfodoc = Path('{}-stage-{}.json'.format(base_name, stage_number))
        pakainfodoc.touch(exist_ok=True)
        with open('{}-stage-{}.json'.format(base_name, stage_number), 'w')  as fp:
            json.dump(save_json, fp)
        
        print("+++++++++++++++++ {} DONE ++++++++++++++++++++++++\n".format(stage_type))
    except:
        c.quit()
###################################################################################
c.quit()
