from os import listdir
from os.path import isfile, join

import numpy
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import chardet
import json
import math
import os
import pickle

from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn import metrics

dataPath = "./data/epl-training.csv"
fifaPath = "./data/fifaRanking.csv"

trainData = pd.read_csv(dataPath)
print("Football-data Dataset Loaded.")

fifaRankings = pd.read_csv(fifaPath)
print("FifaIndex Team Rankigs Dataset Loaded.")

######### LOAD PLAYERS ##########
players_pick = r"./data/players.pickle"
players_folder = r"./data/players"
players = {}
if os.path.exists(players_pick):
    with open(players_pick, 'rb') as handle:
        players = pickle.load(handle)
else:
    players_files = [f for f in listdir(players_folder) if ".json" in f]
    for player_file in players_files:
        player_path = join(players_folder, player_file)
        enc = chardet.detect(open(player_path, 'rb').read())['encoding']
        with open(player_path, encoding=enc) as json_file:
            data = json.load(json_file)
            player_id = data['id']
            players[player_id] = data
            #print(player_path)
    with open(players_pick, 'wb') as handle:
        pickle.dump(players, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Player Dataset Loaded.")

# def getPlayersAndPositions(team, homeOrAway):
#     positions = {}
#     players = {}
#     i = 1
#     for p in team:
#         position = p["matchPosition"]
#         positionColumn = "{homeOrAway}P{index}".format(homeOrAway=homeOrAway, index=position)
#         if positionColumn not in positions:
#             positions[positionColumn] = 0
#         positions[positionColumn] += 1
#         playerColumn = "{homeOrAway}P{index}".format(homeOrAway=homeOrAway, index=i)
#         players[playerColumn] = p["id"]
#         i += 1
#     return positions, players

# positionsAndPlayers = []
# for index, row in trainData.iterrows():
#     path_epl_json = row["Path"][1:]
#
#     enc = chardet.detect(open(path_epl_json, 'rb').read())['encoding']
#     with open(path_epl_json, encoding=enc) as json_file:
#         data = json.load(json_file)
#         team1 = data['teamLists'][0]['lineup']
#         team2 = data['teamLists'][1]['lineup']
#         positions1, players1 = getPlayersAndPositions(team1, "H")
#         positions2, players2 = getPlayersAndPositions(team2, "A")
#         positions = dict(positions1, **positions2)
#         players = dict(players1, **players2)
#         final = dict(players, **positions)
#         positionsAndPlayers.append(final)
#         print(path_epl_json)
# positionsAndPlayers_df = pd.DataFrame(positionsAndPlayers)
# print("Players and positions loaded.")

def extractFifaRanking(teamName, year, ratingType):
    findExact = fifaRankings[(fifaRankings["Team"] == teamName) & (fifaRankings["Year"] == year)]

    if findExact.shape[0] != 1:
        findExact = fifaRankings[(fifaRankings["Team"] == teamName)]
        findExact['Year'] = findExact['Year'].apply(lambda s: abs(s - year))
        findExact = findExact[findExact['Year'] == findExact['Year'].min()]

    return findExact.iloc[0][ratingType]


class PlTeam:
    def __init__(self, name):
        self.name = name

        # Difference between goals scored and goals conceded
        self.goalDiff = 0
        self.gamesPlayed = []
        self.form = 1
        self.fifaRankings = {}

    # Retrieve team's FIFA ranking for season indicated by 'date'
    def getFifaRanking(self, date):
        date = pd.to_datetime(date)
        year = date.year
        # Matches played in autumn of year X belong to the (X + 1) FIFA season
        if date > pd.Timestamp(year, 8, 1):
            year += 1
        if year not in self.fifaRankings:
            self.fifaRankings[year] = extractFifaRanking(self.name, year, "OVR")
        return self.fifaRankings[year]

    # Return float in range [0, 1] indicating whether team has won a lot of games
    # recently (1 if team has won all recent matches, 0 if all lost),
    # weighing more recent games higher than older ones.
    # Modulated by hyperparameter 'k' (how many past matches to take into account).
    # Formula here taken from Baboota & Kaur [1]
    def getWeightedStreak(self, k):
        streak = 0
        matchNo = len(self.gamesPlayed)
        for p in range(matchNo - k, matchNo):
            match = self.gamesPlayed[p]
            result = match["FTR"]
            resultValue = 1
            if match["HomeTeam"] == self.name:
                if result == 'H':
                    resultValue = 3
                elif result == 'A':
                    resultValue = 0
            elif match["AwayTeam"] == self.name:
                if result == 'A':
                    resultValue = 3
                elif result == 'H':
                    resultValue = 0
            weight = p - (matchNo - k - 1)
            normFactor = 3 * (k * (k + 1) / 2)
            streak += (weight * resultValue / normFactor)
        return streak

    def getMeanGCS(self, k):
        matchNo = len(self.gamesPlayed)
        corners = 0
        shotsOnTarget = 0
        goals = 0
        for p in range(matchNo - k, matchNo):
            match = self.gamesPlayed[p]
            if match["HomeTeam"] == self.name:
                corners += match["HC"]
                shotsOnTarget += match["HST"]
                goals += match["FTHG"]
            else:
                corners += match["AC"]
                shotsOnTarget += match["AST"]
                goals += match["FTAG"]

        return goals / k, corners / k, shotsOnTarget / k


# Return tuple of updated 'Form' values (float, float) of participating teams
# to take into account the 'result' of a given match.
# Form indicates the overall ranking of a team relative to others at a given
# time, based on which other teams they've won against and the Form of these
# opponents in turn.
# With each match, the winning team "steals" a fraction of the loser's Form.
# (This fraction is the hyperparameter gamma.)
# Formula taken from Baboota & Kaur [1]
def getUpdatedForm(result, homeForm, awayForm, gamma):
    if result == 'D':
        newHomeForm = homeForm - (gamma * (homeForm - awayForm))
        newAwayForm = awayForm - (gamma * (awayForm - homeForm))
    elif result == 'H':
        newHomeForm = homeForm + (gamma * awayForm)
        newAwayForm = awayForm - (gamma * awayForm)
    else:
        newAwayForm = awayForm + (gamma * homeForm)
        newHomeForm = homeForm - (gamma * homeForm)
    return (newHomeForm, newAwayForm)


def iterateTeams(data):
    for _, row in data.iterrows():
        yield row["HomeTeam"]
        yield row["AwayTeam"]


# Return dictionary of {team name string -> empty PlTeam object} for each
# team in data set.
def getTeams(data):
    return {name: PlTeam(name) for name in set(iterateTeams(data))}


# Returns only games in 'data' played during season indicated by 'seasonNo',
# where season 0 is the 2005/2006 season, 1 is 2006/2007, etc.
def filterSeason(data, seasonNo):
    return data[data["Season"] == seasonNo]


# Constructs a list of dictionaries (each dict of form {teamName -> goalDiff}),
# where indices to the list is the season number (0 -> 2000/2001, 1 -> 2001/2002, etc.)
def getGoalDiffList(data, seasonsCount):
    goalDiffList = []
    for i in range(seasonsCount):
        seasonData = filterSeason(data, i)
        teams = getTeams(seasonData)
        for _, row in seasonData.iterrows():
            homeTeam = teams[row["HomeTeam"]]
            awayTeam = teams[row["AwayTeam"]]
            homeTeam.goalDiff += row["FTHG"] - row["FTAG"]
            awayTeam.goalDiff += row["FTAG"] - row["FTHG"]
        seasonDict = {}
        for team in teams.values():
            seasonDict[team.name] = team.goalDiff
        goalDiffList.append(seasonDict)
    return goalDiffList


################## GAME DIFFERENCE DIAGRAM #####################
# goalDiffList = getGoalDiffList(trainData, 20)
# winners = []
# winners.append(goalDiffList[5]["Chelsea"])
# winners.append(goalDiffList[6]["Man United"])
# winners.append(goalDiffList[7]["Man United"])
# winners.append(goalDiffList[8]["Man United"])
# winners.append(goalDiffList[9]["Chelsea"])
# winners.append(goalDiffList[10]["Man United"])
# winners.append(goalDiffList[11]["Man City"])
# winners.append(goalDiffList[12]["Man United"])
# winners.append(goalDiffList[13]["Man City"])
# winners.append(goalDiffList[14]["Chelsea"])
# winners.append(goalDiffList[15]["Leicester"])
# winners.append(goalDiffList[16]["Chelsea"])
# winners.append(goalDiffList[17]["Man City"])
# plt.bar([2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018], winners)
# plt.show()
#######################################################

################## GAME DIFFERENCE vs Rating ##########
# goalDiffList = getGoalDiffList(trainData, 20)
# season = goalDiffList[17]
# season2 = goalDiffList[16]
# goalDifferences = []
# rankings = []
# for team in season:
#     goalDifferences.append(season[team])
#     rankings.append(extractFifaRanking(team, 2018, "OVR"))
# for team in season2:
#     goalDifferences.append(season2[team])
#     rankings.append(extractFifaRanking(team, 2017, "OVR"))
# plt.scatter(goalDifferences, rankings)
# plt.xlabel("Goal Difference")
# plt.ylabel("Fifa Rating")
#######################################################

###################### HOME ADVANTAGE ########################
count = {'H': 0, 'D': 0, 'A': 0, }

for _, row in trainData.iterrows():
    result = row["FTR"]
    count[result] = count[result] + 1

totalGames = count['H'] + count['A'] + count['D']
homePercent = round(count['H'] / totalGames * 100, 2)
awayPercent = round(count['A'] / totalGames * 100, 2)
drawPercent = round(count['D'] / totalGames * 100, 2)

homeLabel = 'Home' + " - " + str(homePercent) + '%'
awayLabel = 'Away' + " - " + str(awayPercent) + '%'
drawLabel = 'Draw' + " - " + str(drawPercent) + '%'

plt.title("Match result distribution")
plt.bar([homeLabel, drawLabel, awayLabel], height=[count['H'], count['D'], count['A']])
############################################################

def getTeamSkillsFromRow(team, row):
    teamStats = {"Attack": [], "Defence": [], "Goalkeeping": [], "Team Play": []}
    for i in range(1, 12):
        playerColumn = "{homeOrAway}P{index}".format(homeOrAway=team, index=i)
        playerId = row[playerColumn]
        playerData = players[playerId]
        skills = playerData["skills"]
        position = playerData["matchPosition"]

        if position == "G":
            teamStats["Goalkeeping"].append(skills["Goalkeeping"])
            teamStats["Defence"].append(skills["Defence"])
        elif position == "F":
            teamStats["Attack"].append(skills["Attack"])
            teamStats["Team Play"].append(skills["Team Play"])
        elif position == "M":
            teamStats["Attack"].append(skills["Attack"])
            teamStats["Team Play"].append(skills["Team Play"])
        elif position == "D":
            teamStats["Defence"].append(skills["Defence"])
            teamStats["Team Play"].append(skills["Team Play"])
    return teamStats

def getSkillCategoryDataFrame(skills, category, filterColumns):
    df = pd.DataFrame(skills[category])[filterColumns]\
        .replace("%", "", regex=True)\
        .replace(",", "", regex=True)\
        .replace(0, np.nan)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(df)
    imputer.fit(df)
    transformed = imputer.transform(df)

    return pd.DataFrame(transformed, columns=df.columns)

def getAttackSkills(skills):
    attackSkills = getSkillCategoryDataFrame(skills, "Attack", ["Goals", "Goals per match", "Shooting accuracy %"])
    return attackSkills.mean()

def getDefenceSkills(skills):
    defenceSkills = getSkillCategoryDataFrame(skills, "Defence", ["Clean sheets"])
    return defenceSkills.mean()

def getGoalkeepingSkills(skills):
    goalkeepingSkills = getSkillCategoryDataFrame(skills, "Goalkeeping", ["Saves"])
    return goalkeepingSkills.mean()

def getTeamPlaySkills(skills):
    teamPlaySkills = getSkillCategoryDataFrame(skills, "Team Play", ["Assists", "Passes"])
    return teamPlaySkills.mean()

def getTeamSkills(team, row):
    skills = getTeamSkillsFromRow(team, row)

    a = getAttackSkills(skills)
    d = getDefenceSkills(skills)
    g = getGoalkeepingSkills(skills)
    t = getTeamPlaySkills(skills)

    return pd.concat([a, d, g, t], axis=0)

def constructFeatures(data, k, gamma):
    teams = getTeams(data)
    x_data = []
    y_train = []
    for index, row in data.iterrows():
        homeTeam = teams[row["HomeTeam"]]
        awayTeam = teams[row["AwayTeam"]]

        # Ignore each team's first k matches in data
        minStreak = min(len(homeTeam.gamesPlayed), len(awayTeam.gamesPlayed))
        if minStreak >= k:
            homeMeanGoals, homeMeanCorners, homeMeanTargetShots = homeTeam.getMeanGCS(k)
            awayMeanGoals, awayMeanCorners, awayMeanTargetShots = awayTeam.getMeanGCS(k)
            meanGoalsDiff = homeMeanGoals - awayMeanGoals
            meanCornersDiff = homeMeanCorners - awayMeanCorners
            meanTargetShotsDiff = homeMeanTargetShots - awayMeanTargetShots

            goalDiffDiff = homeTeam.goalDiff - awayTeam.goalDiff
            streakDiff = homeTeam.getWeightedStreak(k) - awayTeam.getWeightedStreak(k)
            formDiff = homeTeam.form - awayTeam.form
            fifaRankDiff = homeTeam.getFifaRanking(row["Date"]) - awayTeam.getFifaRanking(row["Date"])

            forwardsDiff = row["HPF"] - row["APF"]
            midfieldDiff = row["HPM"] - row["APM"]
            defendersDiff = row["HPD"] - row["APD"]

            # Transform match result to number
            # -1 -> HOME
            #  0 -> DRAW
            #  1 -> AWAY
            result = row['FTR']
            resultNumber = -1
            if result == "D":
                resultNumber = 0
            elif result == "A":
                resultNumber = 1

            features = [goalDiffDiff, streakDiff, formDiff, fifaRankDiff,
                           meanGoalsDiff, meanCornersDiff, meanTargetShotsDiff,
                           forwardsDiff, midfieldDiff, defendersDiff]

            homeSkills = getTeamSkills("H", row)
            awaySkills = getTeamSkills("A", row)

            skillsDiffVals = (homeSkills - awaySkills).values

            x_data.append(numpy.concatenate([features, skillsDiffVals]))
            y_train.append(resultNumber)

        homeTeam.gamesPlayed.append(row)
        awayTeam.gamesPlayed.append(row)

        # Update goal differentials with goal data from this match
        homeTeam.goalDiff += row["FTHG"] - row["FTAG"]
        awayTeam.goalDiff += row["FTAG"] - row["FTHG"]

        # Update form with final result of this match
        homeTeam.form, awayTeam.form = getUpdatedForm(row["FTR"], homeTeam.form, awayTeam.form, gamma)
        print(index)
    return (x_data, y_train, teams)

def numberToResult(result):
  if result == -1:
    return "Home"
  elif result == 1:
    return "Away"
  else:
    return "Draw"


###################### TRAIN MODEL AND TEST ############################
#imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(trainData)
#imputer.fit(df)
#transformed = imputer.transform(df)
trainDataComputedPath = r"data/trainData.pickle"
if os.path.exists(trainDataComputedPath):
    with open(trainDataComputedPath, 'rb') as handle:
        x_train, y_train = pickle.load(handle)
else:
    trainData = trainData.dropna(subset=["AP11", "HP11", "HPF", "APF"])
    x_train, y_train, _ = constructFeatures(trainData, 6, 0.33)
    with open(trainDataComputedPath, 'wb') as handle:
        pickle.dump((x_train, y_train), handle, protocol=pickle.HIGHEST_PROTOCOL)


# team player lowers the score 16
# 4, 5, 6 means dont affect score
# 7, 8, 9 team structure affect
x_train = np.asarray(x_train, dtype=np.float32)[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]

#################### Pearson correlation ############################
# import seaborn as sns
# plt.figure(figsize=(20,10))
# sns.heatmap((pd.DataFrame(x_train)).corr(), annot= True)
######################################################################

normalizer = preprocessing.StandardScaler().fit(x_train)
x_train = normalizer.transform(x_train)

i = (int)(len(x_train) / 5 * 4)
trainSet = x_train[:i]
testSet = x_train[i:]
trainResult = y_train[:i]
testResult = y_train[i:]

finalClf = SVC(kernel='rbf', C=100, gamma=0.00001).fit(trainSet, trainResult)
print(classification_report(trainResult, finalClf.predict(trainSet)))
print(classification_report(testResult, finalClf.predict(testSet)))
###############################################################################

####################### CONFUSION MATRIX ######################################
    # from sklearn.metrics import confusion_matrix
    # import itertools
    #
    # y_true = list(map(numberToResult, trainResult))
    # y_pred = list(map(numberToResult, finalClf.predict(trainSet)))
    #
    # cm = confusion_matrix(y_true, y_pred)
    # classes = ["Away", "Draw", "Home"]
    #
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix")
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    #
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], 'd'),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
#############################################################################

########################## FETURE VERIFICATION ##############################
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(trainSet, trainResult)
# print(model.feature_importances_)
# pd.Series(model.feature_importances_).plot.bar(color='steelblue', figsize=(12, 6), rot=0)
#pd.Series(model.feature_importances_,["Goal Difference", "Streak", "Form", "Fifa Ranking"]).plot.bar(color='steelblue', figsize=(12, 6), rot=0)