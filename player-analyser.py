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
import chardet
import json

players_folder_new = r""
players_files = [f for f in listdir(players_folder_new) if ".json" in f]
topStatHeaders = set()
statHeaders = set()

for player_file in players_files:
    player_path = join(players_folder_new, player_file)

    enc = chardet.detect(open(player_path, 'rb').read())['encoding']
    with open(player_path, encoding=enc) as json_file:
        data = json.load(json_file)

        for k in data['topStats'].keys():
            topStatHeaders.add(k)
        for k in data['skills'].keys():
            statHeaders.add(k)

print(statHeaders)
print(topStatHeaders)