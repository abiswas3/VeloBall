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

import crawler
import parsers
import parsers.utils as utils
from tabulate import tabulate


with open('researchers.json') as fp:
    WEB_LINK = json.load(fp)

c = crawler.GoogleScholarHTMLRenderer()    
for (researcher, url) in WEB_LINK.items():
    ######################################
    print(researcher, url)
    fname = '../saved_htmls/{}'.format(utils.get_hash(url))
    extractor = None
    c.get_main_page(url, fname, override=False)
    ######################################

    extractor = parsers.GoogleScholar()
    with open(fname) as fp:
        temp = json.load(fp)
        url = temp['url']
        page = temp['page']
        meta_data = {}
        nlp = None

    extractor.process_page(url, page, schema_org_check=False, urls_crawled=[])
    # DO a diff and print
    
    with open('papers/{}.json'.format(researcher), 'w') as fp:
        json.dump(extractor.papers.to_json(), fp)

    print(tabulate(extractor.papers.head(20), headers='keys', tablefmt='psql'))
    print('\n\n')

c.quit()

