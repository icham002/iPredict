from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import os
import datetime
import pandas as pd
import time
import re
import requests
import pickle
from os import listdir
from os.path import isfile, join
import sys
import html
from webdriver_manager.chrome import ChromeDriverManager

phantomjs_path = r"C:\Python38\misc\phantomjs.exe"
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
browser = webdriver.Chrome(ChromeDriverManager().install(), options=options)

link_template = "https://www.premierleague.com/match/"
path_matches = r"PathToSave"

index_curr = 48000
index_end = 47000

while index_curr >= index_end:
    link = link_template + str(index_curr)
    match_path = r"{path}\{name}.json".format(path=path_matches, name=index_curr)

    if os.path.exists(match_path):
        index_curr = index_curr - 1
        continue

    try:
        browser.get(link)
        time.sleep(0.2)

        try:
            json = browser.find_element_by_class_name("mcTabsContainer").get_attribute("data-fixture")

            json_file = open(r"{path}\{name}.json".format(path=path_matches, name=index_curr), "w")
            json_file.write(json)
            json_file.close()

            print(r"match {index} saved".format(index=index_curr))
        except Exception as e:
            print(r"ERROR match {index} ERROR".format(index=index_curr))
            print(e, flush=True)
    except Exception as e:
        print(r"ERROR match {index} ERROR".format(index=index_curr))
        print(e, flush=True)

    index_curr = index_curr - 1
print("DONE")