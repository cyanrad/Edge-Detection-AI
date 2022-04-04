
import pandas as pd
import random as rand

column_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'edge']
raw_dataset = pd.read_csv('data.csv', names=column_names)
dataset = raw_dataset.copy()
print(dataset['edge'].value_counts())

zeroesToDelete = dataset['edge'].value_counts()[0] - \
    dataset['edge'].value_counts()[1]
print(zeroesToDelete)

for i in range(0, int(zeroesToDelete)):
    print(i)
    rowToDel = rand.randint(0, zeroesToDelete-i)
    if(dataset['edge'].values[rowToDel] == 0):  # if the row is edge
        try:
            dataset.drop(rowToDel, axis=0, inplace=True)
        except KeyError:
            print(KeyError)
            dataset.to_csv("fiftySplit.csv")
    else:   # else the loop doesn't count
        i = i-1


dataset.to_csv("fiftySplit.csv")
