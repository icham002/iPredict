import os
from os import listdir
from os.path import join
import json
import glob
import pandas as pd
import datetime
from unidecode import unidecode
import editdistance

matches = pd.read_csv(r"matches_linked.csv")['Path']

players_cache = set()
players_folder = r"pathtosaveplayers"

def getPlayerStructure(p):
    country = ""
    try:
        country = p['birth']['country']['country']
    except Exception as e:
        country = p['birth']['country']['isoCode']

    return {'name': unidecode(p['name']['display']),
            'country': country,
            'birthday': datetime.datetime.strptime(p['birth']['date']['label'], "%d %B %Y")}

def processTeam(team):
    for p in team:
        id = p['id']
        if id not in players_cache:
            players_cache.add(id)
            #players.append(getPlayerStructure(p))
            file_path = "{folder}\\{id}.json".format(folder=players_folder, id=p['id'])
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as outfile:
                    print(file_path)
                    json.dump(p, outfile, ensure_ascii=False)

for match in matches:
    with open(match) as json_file:
        data = json.load(json_file)
        home_team = data['teamLists'][0]['lineup']
        away_team = data['teamLists'][1]['lineup']

        processTeam(home_team)
        processTeam(away_team)

# i_z = 0
# i_m = 0
# i_o = 0
# for player_epl in players:
#     #for player_db in players_df:
#     #found = players_df[players_df['Birth_Date'] == player_epl['birthday'].strftime('%m/%d/%Y')]
#     found_count = 0
#     for i, player_db in players_df.iterrows():
#         dist = editdistance.eval(player_db['Name'], player_epl['name'])
#         if dist <= 3:
#             found_count = 1
#             break
#
#     if found_count == 1:
#         print(player_db['Name'], player_epl['name'])
#     else:
#         print ('damn')

