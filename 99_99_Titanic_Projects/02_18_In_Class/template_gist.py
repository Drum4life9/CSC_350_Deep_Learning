import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from d2l import torch as d2l

import model_code
import seaborn as sns
import torch
from torch import nn
import tensorflow as tf
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# matplotlib.use('TkAgg')

from numpy.random import seed

# To ignore warinings
import warnings
warnings.filterwarnings('ignore')

# Boolean to show plots & diagnostics
show_plots = False
show_diagnostics = False

train_data = pd.read_csv("../99_99_data/train.csv")
# print(train_data.head())
df = train_data

test_data = pd.read_csv("../99_99_data/test.csv")
# print(test_data.head())

if show_diagnostics:
    print(df.info())


###   Data Processing - Ages  ###
# Fill in missing ages based on the title in the passenger's name
# create new Title column
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
if show_diagnostics:
    print(df.head())
    print(df['Title'].value_counts())  # how many titles are there?

# Use the seven most common titles (Mr, Miss, Mrs, Master, Rev, Dr)
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
           'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
           'Capt': 'Mr', 'Countess': 'Mrs', 'Ms':'Miss',  'Dona': 'Mrs'}
df.replace({'Title': mapping}, inplace=True)
if show_diagnostics:
    print(df['Title'].value_counts())

# Use the median of each title group to fill in missing data
title_ages = dict(df.groupby('Title')['Age'].median())

# create a column of the average ages
df['age_med'] = df['Title'].apply(lambda x: title_ages[x])

# replace all missing ages with the value in this column
df['Age'].fillna(df['age_med'], inplace=True, )
del df['age_med']

if show_plots:
    sns.barplot(x='Title', y='Age', data=df, estimator=np.median, ci=None, palette='Blues_d')
    plt.xticks(rotation=45)
    plt.show()

if show_plots:
    sns.countplot(x='Title', data=df, palette='hls', hue='Survived')
    plt.xticks(rotation=45)
    plt.show()


###   Data Processing - Embarked  ###
# Fill in the missing ports with the mode of the ports
freq_port = df.Embarked.dropna().mode()[0]
if show_diagnostics:
    print(f"most frequent port is {freq_port}")  # Gives us 'S'

df['Embarked'] = df['Embarked'].fillna(freq_port)


###   Data Processing - Family Size  ###
# We can create a new column called Family_Size
df['Family_Size'] = df['Parch'] + df['SibSp']


###  Encode Categorical Variables  ###
# Keeping categorical variables: Embarked, Sex, and Title

# convert to category dtype
df['Sex'] = df['Sex'].astype('category')
# convert to category codes
df['Sex'] = df['Sex'].cat.codes
# print(df['Sex'])

# subset all categorical variables which need to be encoded
categorical = ['Embarked', 'Title']

for var in categorical:
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]

# drop the variables we won't be using
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
# print(df.head())
# print(df.iloc[0])


# ###  A non-NN model  ###
train = df[pd.notnull(df['Survived'])]

# create a validation set
X_train, X_val, y_train, y_val = train_test_split(train.drop(['Survived'], axis=1),
    train['Survived'], test_size=0.001, random_state=42)

X_train = X_train.astype({col: 'int8' for col in train.select_dtypes('bool').columns})


# if show_diagnostics:
#     for i in [X_train, X_val]:
#         print(i.shape)
#
# # Random Forest Model
# # random_state is set to 42 for reproducibility
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)
# print(accuracy_score(y_val, rf.predict(X_val)))

data = torch.tensor(X_train.values, dtype = torch.float32)

if __name__ == '__main__':
    titanic_model = model_code.get_in_class_titanic(X_train)
    training = titanic_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    val_acc = np.mean(training.history['val_accuracy'])
    print("\n%s: %.2f%%" % ('val_accuracy', val_acc * 100))

    plt.plot(training.history['accuracy'], label='accuracy')
    plt.plot(training.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.show()
