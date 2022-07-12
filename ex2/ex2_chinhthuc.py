import pandas as pd
import numpy as np
import csv
import math

print('Analytics - 1: Compute average critics rating for each film ')
csv_columns = ['Movie', 'Release Date', 'Critic 1', 'Critic 2', 'Critic 3', 'Critic 4', 'Critic 5', 'Audience Rating']
df = pd.read_csv('data.csv', usecols=csv_columns)
# set initial value for max_value and min value of distribution
max_value = int(df['Critic 1'][0])
min_value = int(df['Critic 1'][0])
all_value = []


def calc_average(a, b, c, d, e):
    global max_value
    global min_value
    avg = []
    if a != '':
        a = int(a)
        avg.append(a)
        if a > max_value:
            max_value = a
        if a < min_value:
            min_value = a

    if b != '':
        b = int(b)
        avg.append(b)
        if b > max_value:
            max_value = b
        if b < min_value:
            min_value = b

    if c != '':
        c = int(c)
        avg.append(c)
        if c > max_value:
            max_value = c
        if c < min_value:
            min_value = c

    if d != '':
        d = int(d)
        avg.append(d)
        if d > max_value:
            max_value = d
        if d < min_value:
            min_value = d

    if e != '':
        e = int(e)
        avg.append(e)
        if e > max_value:
            max_value = e
        if e < min_value:
            min_value = e

    avg_value = sum(avg) / len(avg)
    return avg_value


with open('data.csv', 'r') as csvinput:
    with open('output.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)
        all = []
        row = next(reader)
        row.append('Average Critics Rating')
        all.append(row)
        for row in reader:
            avg_value = calc_average(row[2], row[3], row[4], row[5], row[6])
            row.append(avg_value)
            all.append(row)
        writer.writerows(all)
print('Done Analytics - 1: Open output.csv to see column Average critics rating  ')
print("-------------------------------------------------------------------------------------")
print("Modeling - 1.1 : Normalize the values for the critics to [0:1]")
for index, row in df.iterrows():
    if pd.notna(row[2]):
        value_append = (int(row[2]) - min_value) / (max_value - min_value)
        all_value.append(value_append)
        df.loc[index, ['Critic 1']] = [value_append]
    if pd.notna(row[3]):
        value_append = (int(row[3]) - min_value) / (max_value - min_value)
        all_value.append(value_append)
        df.loc[index, ['Critic 2']] = [value_append]
    if pd.notna(row[4]):
        value_append = (int(row[4]) - min_value) / (max_value - min_value)
        all_value.append(value_append)
        df.loc[index, ['Critic 3']] = [value_append]
    if pd.notna(row[5]):
        value_append = (int(row[5]) - min_value) / (max_value - min_value)
        all_value.append(value_append)
        df.loc[index, ['Critic 4']] = [value_append]
    if pd.notna(row[6]):
        value_append = (int(row[6]) - min_value) / (max_value - min_value)
        all_value.append(value_append)
        df.loc[index, ['Critic 5']] = [value_append]
print("DataFrame after first Normalize")
print(df[['Critic 1', 'Critic 2', 'Critic 3', 'Critic 4', 'Critic 5']])

print("Modeling - 1.1 :  transform values to z-scores")

mean = sum(all_value) / len(all_value)
square_distance = 0
for x in all_value:
    square_distance = square_distance + pow(x - mean, 2)

standard_deviation = math.sqrt(square_distance / len(all_value))
all_z_score = []

for index, row in df.iterrows():
    z_score_array = []
    if pd.notna(row[2]):
        z_score = (float(row[2]) - mean) / standard_deviation
        z_score_array.append(z_score)
        df.loc[index, ['Critic 1']] = [z_score]
    if pd.notna(row[3]):
        z_score = (float(row[3]) - mean) / standard_deviation
        z_score_array.append(z_score)
        df.loc[index, ['Critic 2']] = [z_score]
    if pd.notna(row[4]):
        z_score = (float(row[4]) - mean) / standard_deviation
        z_score_array.append(z_score)
        df.loc[index, ['Critic 3']] = [z_score]
    if pd.notna(row[5]):
        z_score = (float(row[5]) - mean) / standard_deviation
        z_score_array.append(z_score)
        df.loc[index, ['Critic 4']] = [z_score]
    if pd.notna(row[6]):
        z_score = (float(row[6]) - mean) / standard_deviation
        z_score_array.append(z_score)
        df.loc[index, ['Critic 5']] = [z_score]
    if len(z_score_array) == 5:
        all_z_score.append(z_score_array)
print("Data Frame after  transform values to z-scores")
print(df[['Critic 1', 'Critic 2', 'Critic 3', 'Critic 4', 'Critic 5']])
print("-------------------------------------------------------------------------------------")
print("Modeling - 1.2 :  Compute the mean")
all_z_score = np.array(all_z_score)
all_mean = []
for col in range(all_z_score.shape[1]):
    print("+++++")
    print(all_z_score[:, col])
    all_mean.append(np.sum(all_z_score[:, col])/len(all_z_score[:, col]))
print("Here is 5 means of 5 Critic")
print(all_mean)
#mean = (sum(map(sum, all_z_score))) / (len(all_z_score) * len(all_z_score[0]))
# print(mean)
unit_matrix = np.ones((len(all_z_score), len(all_z_score)))

Deviation_Matrix = all_z_score - 5 * np.dot(unit_matrix, all_z_score)
print("Modeling - 1.2 : Covariance Matrix")
Covariance_Matrix = np.dot(Deviation_Matrix.T, Deviation_Matrix)
print(Covariance_Matrix)
all_SD = []
for i in range(len(Covariance_Matrix)):
    all_SD.append(math.sqrt(Covariance_Matrix[i][i]))
print("Modeling - 1.2 : All standard deviation")
print(all_SD)
print("-------------------------------------------------------------------------------------")
print("Modeling - 1.3 : All standard deviation")

