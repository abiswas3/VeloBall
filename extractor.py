from lxml import etree
from io import StringIO
import extruct
import json
import string
from collections import Counter
import numpy as np
import pathlib

from . import utils
from . import dom_node 

class DocumentExtractor():
    
    def __init__(self, **kwargs):

        self.parser = etree.HTMLParser()
        self.nlp = None
        if 'spacy_import' in kwargs:
            self.nlp = 1 # change this to real spacy

        self.meta_data = {}
        if 'meta_data' in kwargs:
            self.meta_data = kwargs['meta_data']

        
    def process_page(self,
                     url,
                     page,
                     schema_org_check=True,
                     **kwargs):
        
        """FIXME! briefly describe function

        :param url: 
        :param page: 
        :param schema_org_check: 
        :returns: 
        :rtype: 

        """

        self.bad_page = False
        if schema_org_check:
            try:
                schema_org = extruct.extract(page, base_url=url)
                if len(schema_org['json-ld']):
                    self.schema_org = schema_org
            except:
                pass
        
        # GET ETREE
        # This could raise an exception (NOTE)
        try:
            self.tree   = etree.parse(StringIO(page), self.parser)
        except:
            self.bad_page = True
            
        self.curr_page = page
        self.curr_url = url

        text_nodes = [dom_node.Text_Node(x, x.getparent(), self.getpath())
                      for x in self.tree.xpath("//text()")]
        self.text_nodes = [node for node in text_nodes if node.valid]

        # Some fuzzy title node calcullations
        title_node = [node for node in self.text_nodes if node.is_title]
        if len(title_node) > 1:
            temp = np.argsort([len(node.text) for node in title_node])[::-1]
            self.title_node = title_node[temp[0]]

        elif len(title_node) == 1:
            self.title_node = title_node[0]
        else:
            self.title_node = None

            
    def save_minimum_info(self, base=""):
        
        fname = utils.get_hash(self.curr_url)
        if len(base):
            if base[-1] == '/':
                base = base[:-1]

        pathlib.Path(base).mkdir(parents=True,
                                 exist_ok=True)

        with open("{}/{}".format(base, fname), 'w') as fp:
            json.dump({"url": self.curr_url,
                       "page": self.curr_page},
                      fp)

    def load_object(self, fname):

        with open(fname) as fp:
            temp = json.load(fp)

        self.curr_url = temp["url"]
        self.curr_page = temp["page"]

        
    def getpath(self):
        """Returns a function that can extract xpath of a lxml.etree node
        :returns: See above
        :rtype: function

        """
        return self.tree.getpath

    def get_links(self):
        """Get all hrefs in a webpage

        :returns: A generator of lxml.etree elements with tag <a>
        :rtype: generator

        """

        return self.tree.xpath("//a[@href]")

    def get_imgs(self):

        """Get all imgs in a webpage

        :returns: A generator of lxml.etree elements with tag <img>
        :rtype: generator

        """

        return self.tree.xpath("//img")
