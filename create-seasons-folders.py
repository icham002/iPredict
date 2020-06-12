import os
from os import listdir
from os.path import join
import json
import shutil

path_matches = r""
path_epl_matches = r""
matches = [f for f in listdir(path_matches) if ".json" in f]
players = set()

for match in matches:
    match_file = join(path_matches, match)
    with open(match_file) as json_file:
        data = json.load(json_file)
        try:
            season = data['gameweek']['compSeason']['label'].replace("/", "-")
            competition = data['gameweek']['compSeason']['competition']['description']

            if competition == "Premier League" and not season.contains("19"):
                season_path = join(path_epl_matches, season)
                if not os.path.exists(season_path):
                    os.makedirs(season_path)
                path_copy = join(season_path, match)
                if not os.path.exists(path_copy):
                    shutil.copyfile(match_file, path_copy)
                    print(path_copy)
        except Exception as e:
            print(e)