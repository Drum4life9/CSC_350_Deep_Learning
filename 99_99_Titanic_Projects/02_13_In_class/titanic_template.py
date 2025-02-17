import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("../99_99_data/train.csv")
test_data = pd.read_csv("../99_99_data/test.csv")
# print(train)

def preprocess_name_ticket(df):
    df = df.copy()

    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

    def ticket_number(x):
        return x.split(" ")[-1]

    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])

    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    return df


train = preprocess_name_ticket(train)

print(pd.crosstab(train.Survived, train.Pclass))
# plt.scatter(train.Survived, train.Fare)
count_survived = train.Survived.value_counts()
count_survived = [count_survived[0], count_survived[1]]

# plt.show()
print("---------")

for label in train.columns:
    num_data = train[label].isna().sum()
    print(train[label].value_counts())
    print("nulls: ", num_data)
    print("-----\n")

women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

sns.countplot(train, x="Pclass", hue="Survived", stat = "percent")
plt.show()

sns.countplot(train, x="Sex", hue="Survived", stat = "percent")
plt.show()

sns.countplot(train, x="Age", hue="Survived", stat = "percent")
plt.show()

train.Sex = [1 if i=="male" else 0 for i in train.Sex]

'''
Notes / things to do:

1) Remove name column, or split it up
2) Maybe remove ticket number?

Age has 177 nulls
Cabin has 687 nulls
Embarked has 2 nulls


'''