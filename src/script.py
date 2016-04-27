import pandas as pd
import matplotlib.pyplot as plt

##############################################################################
#                         Loading and preparing data                         #
##############################################################################

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

outcome = train.Survived
outcome.value_counts(normalize=True)

train.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)

train['Test'] = False
test['Test'] = True

data = pd.concat([train, test], ignore_index=True)

##############################################################################
#                               Data wrangling                               #
##############################################################################

data.info()
data.isnull().sum()

# Sex
data['Male'] = (data.Sex == 'male')
data.drop('Sex', axis=1, inplace=True)

# Name
data['Title'] = data.Name.str.split(',').str[1].str.split().str[0]
freqTitles = set(data.Title.value_counts()[:4].index)
data['Title'] = data.Title.apply(lambda x: x if x in freqTitles else 'Rare')
data.drop('Name', axis=1, inplace=True)

# Pclass
data.Pclass.value_counts().plot('bar')

# SibSp
data.SibSp.value_counts().plot('bar')

# Parch
data.Parch.value_counts().plot('bar')
