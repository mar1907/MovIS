import argparse
import pandas as pd
from datetime import datetime
import tensorflow as tf
import ast
import movie_data

DNN = movie_data.DNN

# get data
data = pd.read_csv("tmdb_5000_movies.csv")

# separate numerical data
numericaldata = data[['budget', 'popularity', 'revenue', 'runtime', 'vote_count']]

# transform release date in numerical data
date_format = '%Y-%m-%d'
today = datetime.strptime('2018-01-01', date_format)
timedata = data['release_date'].apply(lambda x: (today - datetime.strptime(str(x) if str(x) != 'nan' else "1920-01-01",
                                                                           date_format)).days)

numericaldata = pd.concat([numericaldata, timedata], axis=1)


# transform genres in numerical data
genres = {}
for line in data['genres']:
    for gen in ast.literal_eval(line):
        genres[gen['id']] = gen['name'].replace(" ", "")

gendata = pd.DataFrame(columns=genres.values(), index=data.index.tolist())
j = 0
for line in data['genres']:
    vect = list(genres.keys())
    genr = {}
    for gen in ast.literal_eval(line):
        genr[gen['id']] = gen['name'].replace(" ", "")
    for i in range(0, len(vect)):
        if vect[i] in genr.keys():
            vect[i] = 1
        else:
            vect[i] = 0
    gendata.iloc[j] = vect
    j += 1

numericaldata = pd.concat([numericaldata, gendata], axis=1)


# apply normalization and add epsilon to data
divide = numericaldata.max() - numericaldata.min()

numericaldata = numericaldata / divide

epsilon = 1e-8
numericaldata = numericaldata + epsilon


# add the feature to be predicted
if DNN:
    ratingdata = data['vote_average'].round(0).astype(int)  # dnn classifier
else:
    ratingdata = data['vote_average']   # linear regression
numericaldata = pd.concat([numericaldata, ratingdata], axis=1)


# replace nan's
numericaldata = numericaldata.fillna(1e-8)

# shuffle
numericaldata = numericaldata.sample(frac=1)

# save data
numericaldata[:4000].to_csv("scaled_movies_train.csv", encoding="utf-8")
numericaldata[4001:].to_csv("scaled_movies_validate.csv", encoding="utf-8")
numericaldata[4001:].to_csv("scaled_movies_test.csv", encoding="utf-8")
