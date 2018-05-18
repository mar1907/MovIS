import ast
from datetime import datetime

import pandas as pd

import correlated_movie_data

tdata = pd.read_csv("tmdb_5000_movies.csv")

idata = pd.read_csv("IMDB-Movie-Data.csv")
idata = pd.concat([idata['Title'], idata['Rating']], axis=1)

titles = tdata['original_title'].str

returndata = pd.DataFrame(columns=['Title', 'Rating'])

for index, row in idata.iterrows():
    if titles.contains(row['Title']).any():
        returndata = returndata.append(row)

tdata = tdata.rename(columns={'original_title': 'Title'})
tdata = tdata.loc[tdata['Title'].isin(idata['Title'])]
tdata = pd.merge(tdata, idata, on='Title')

tdata.to_csv("correlated_movies.csv")

data = tdata

DNN = correlated_movie_data.DNN

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
    # ratingdata = data['vote_average'].round(0).astype(int)  # dnn classifier
    ratingdata2 = data['Rating'].round(0).astype(int)  # dnn classifier
else:
    # ratingdata = data['vote_average']   # linear regression
    ratingdata2 = data['Rating']  # linear regression
numericaldata = pd.concat([numericaldata, ratingdata2], axis=1)


# replace nan's
numericaldata = numericaldata.fillna(1e-8)

# shuffle
numericaldata = numericaldata.sample(frac=1)

# save data
count = int(4*numericaldata['Rating'].count()/5)
numericaldata[:count].to_csv("correlated_movies_train.csv", encoding="utf-8")
numericaldata[count:].to_csv("correlated_movies_validate.csv", encoding="utf-8")
numericaldata[count:].to_csv("correlated_movies_test.csv", encoding="utf-8")
