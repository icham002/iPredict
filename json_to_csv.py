import os
from os import listdir
from os.path import join
import pandas as pd
import json
import datetime

path_epl_matches = r""
csv_path = r""
matches = [join(path_epl_matches, f) for f in listdir(path_epl_matches) if ".csv" in f]

df = []

for folder, subs, files in os.walk(r""):
    for f in files:
        path_epl_json = join(folder, f)
        with open(path_epl_json) as json_file:
            data = json.load(json_file)
            try:
                time = data['kickoff']['millis']
                time = datetime.datetime.fromtimestamp(time/1000.0)
                time = datetime.datetime(time.year, time.month, time.day).timestamp() * 1000

                season = data['gameweek']['compSeason']['label']
                outcome = data['outcome']

                team1_name = data['teams'][0]['team']['name']
                team1_name_short = data['teams'][0]['team']['shortName']
                team1_score = data['teams'][0]['score']

                team2_name = data['teams'][1]['team']['name']
                team2_name_short = data['teams'][1]['team']['shortName']
                team2_score = data['teams'][1]['score']

                halfHomeScore = data['halfTimeScore']['homeScore']
                halfAwayScore = data['halfTimeScore']['awayScore']

                m = {
                    'Date': time,
                    'HomeTeam': team1_name,
                    'HomeTeamShort': team1_name_short,
                    'AwayTeam': team2_name,
                    'AwayTeamShort': team2_name_short,
                    'FTHG': team1_score,
                    'FTAG': team2_score,
                    'HTHG': halfHomeScore,
                    'HTAG': halfAwayScore,
                    'FTR': outcome,
                    'Season': season,
                    'Path': path_epl_json
                }
                df.append(m)
                print(m)
            except Exception as e:
                print(e)

df = pd.DataFrame(df)
df.to_csv(csv_path)
