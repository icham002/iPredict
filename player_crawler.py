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
import chardet
import json

phantomjs_path = r"C:\Python38\misc\phantomjs.exe"
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
browser = webdriver.Chrome(ChromeDriverManager().install(), options=options)

link_template = "https://www.premierleague.com/players/{id}/player/stats"
link = ""

players_folder = r"PlayersPath"
players_files = [f for f in listdir(players_folder) if ".json" in f]

players_folder_new = r"PlayersUpdatedPath"

for player_file in players_files:
    player_path = join(players_folder, player_file)
    player_path_new = join(players_folder_new, player_file)

    if os.path.exists(player_path_new):
        continue

    enc = chardet.detect(open(player_path, 'rb').read())['encoding']
    with open(player_path, encoding=enc) as json_file:
        data = json.load(json_file)
        id = data['id']
        link = link_template.format(id=id)

        try:
            browser.get(link)
            time.sleep(0.1)

            try:
                soup_espn = BeautifulSoup(browser.page_source, features="html.parser")

                topStats = soup_espn.find_all('div', attrs={'class': 'topStat'})
                general = {}
                for topStat in topStats:
                    key = str(topStat.find('span', attrs={'class': 'stat'}).contents[0]).strip()
                    value = topStat.find('span', attrs={'class': 'allStatContainer'}).text.strip()
                    general[key] = value
                data['topStats'] = general

                stats_blocks = soup_espn.find_all('div', attrs={'class': 'statsListBlock'})
                blocks = {}
                for block in stats_blocks:
                    headerStats = block.find('div', attrs={'class': 'headerStat'})
                    header = headerStats.text.replace('\n', '')

                    normalStats = block.find_all('div', attrs={'class': 'normalStat'})

                    stats = {}
                    for stat in normalStats:
                        key = str(stat.find('span', attrs={'class': 'stat'}).contents[0]).strip()
                        value = stat.find('span', attrs={'class': 'allStatContainer'}).text.strip()
                        stats[key] = value
                    blocks[header] = stats
                data['skills'] = blocks

                with open(player_path_new, 'w', encoding='utf-8') as outfile:
                    print(player_path_new)
                    json.dump(data, outfile, ensure_ascii=False)
                print(r"player {index} saved".format(index=id))
            except Exception as e:
                print(r"ERROR player {index} ERROR".format(index=id))
                print(e, flush=True)
        except Exception as e:
            print(r"ERROR player {index} ERROR".format(index=id))
            print(e, flush=True)
