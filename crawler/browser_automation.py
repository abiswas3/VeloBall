'''This module is concerned with all Python related browser
automatiomn. Seleniums limited Python API leads us to write a separate
module to extract visual features in Node.js. This module is used to
load the page and run the required JS to render the page.
'''
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
# from selenium.webdriver.remote.command import Command

# Lambda functions
# get_fontsize = lambda x: x._execute(Command.GET_ELEMENT_VALUE_OF_CSS_PROPERTY,
#                                     {"propertyName": "font-size"})['value']

def make_driver():
    """Creates a selenium webdriver

    :param string choice: Choice of web browser we wish to emulate
    :returns: The webdriver to render webpage with dynamic JS
    :rtype: Selenium.webdriver
    """

    # currently only handles one choice
    return webdriver.Firefox()
