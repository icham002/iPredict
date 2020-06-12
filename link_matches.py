from os import listdir
import os
import pandas as pd
import datetime
import editdistance

folder = r"C:\\EPL\\"
seasons = [f for f in listdir(folder) if ".csv" in f]
columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'FTR']

dfs = []
matches = pd.read_csv(r"")

for s in seasons:
    file_path = os.path.join(folder, s)
    df = pd.read_csv(file_path)
    dfs.append(df)

df_full = pd.concat(dfs)
df_full['Date'] = df_full['Date'].apply(lambda x: datetime.datetime.strptime(x, "%d/%m/%y").timestamp() * 1000)

paths = []
seasons = []

for index, row in df_full.iterrows():
    possibilities = matches[
        (matches['Date'] == row['Date']) &
        (matches['FTHG'] == row['FTHG']) &
        (matches['FTAG'] == row['FTAG']) &
        (matches['HTHG'] == row['HTHG']) &
        (matches['HTAG'] == row['HTAG'])]

    if possibilities.shape[0] == 1:
        optimal_match = possibilities.iloc[0]
    elif possibilities.shape[0] > 1:
        distances = []
        for i, r in possibilities.iterrows():
            e1 = editdistance.eval(r['HomeTeamShort'], row['HomeTeam'])
            e2 = editdistance.eval(r['AwayTeamShort'], row['AwayTeam'])
            distances.append((r, e1 + e2))
        distances.sort(key=lambda distance: distance[1])
        optimal_match = distances[0][0]
        if distances[0][1] > 3:
            print("{a} ({b}) - {c} @@ {d} ({e}) - {f}".format(
               a=distances[0][0]['HomeTeam'],
               b=distances[0][0]['HomeTeamShort'],
               c=row['HomeTeam'],
               d=distances[0][0]['AwayTeam'],
               e=distances[0][0]['AwayTeamShort'],
               f=row['AwayTeam']
            ))
            print()
    else:
        print('ERROR')

    paths.append(optimal_match['Path'])
    seasons.append(optimal_match['Season'])

df_full['Path'] = paths
df_full['Season'] = seasons
print('DONE')