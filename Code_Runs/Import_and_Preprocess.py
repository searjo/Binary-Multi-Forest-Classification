#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 00:01:38 2021

@author: johnsearight
"""

## Import and read the data
"""

train = pd.read_csv('/content/MultiClass_Train.csv', index_col='Index')
test = pd.read_csv('/content/MultiClass_Test.csv', index_col=False)
#Index_col=False prevents pandas from characterizing the first column as Index, and instead creates a new index starting from 0
test = test.drop(columns=['Index'], axis=1)
#Index numbers start around 61006 for test set, so resetting the index to start test at zero using the above method

#Info shows range, column names, non-null values and variable type
train.info()
test.info()

#Head shows first five rows
train.head()
test.head()

#Show mean values of variables
train.mean()
test.mean()

"""## Visualize data

### Visualize categoricals
"""

#Visualize frequency of cover_types. Could have also used the training set for this
cover_types = train.filter(['Cover_Type'], axis=1)
cover_types.value_counts()
sns.countplot(x=cover_types['Cover_Type'], data=cover_types, palette='pastel')

#Pie chart visualization of other categorical variables

#Start with Wilderness Area using value_counts

cmap = sns.color_palette("Set1", as_cmap=True)(np.arange(7))
#Use Set1 to differentiate from other categorical variables
plt.figure(figsize=(10, 10))
plt.pie(train['Wilderness_Area'].value_counts().values,
colors=cmap,
labels=train['Wilderness_Area'].value_counts().keys(),
autopct='%.2f%%')
#To 2 decimal points
plt.title("Wilderness Area Distribution - Training Data")
#plt.show()

cmap = sns.color_palette("Set1", as_cmap=True)(np.arange(7))
#Use Set1 to differentiate from other categorical variables
plt.figure(figsize=(10, 10))
plt.pie(test['Wilderness_Area'].value_counts().values,
colors=cmap,
labels=test['Wilderness_Area'].value_counts().keys(),
autopct='%.2f%%')
#To 2 decimal points
plt.title("Wilderness Area Distribution - Testing Data")
#plt.show()

"""The training and testing sets have a nearly identical make-up of Wilderness_Area.

Note, from data information given at 
https://archive.ics.uci.edu/ml/datasets/Covertype 

"This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices. 

Some background information for these four wilderness areas: Neota (area 2) probably has the highest mean elevational value of the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area 3) would have a lower mean elevational value, while Cache la Poudre (area 4) would have the lowest mean elevational value."

This indicates there must be correlation between wilderness_areas and elevation
"""

#Visualization of Soil_Type in training set using value_counts
cmap = sns.color_palette("Set2", as_cmap=True)(np.arange(7))
#Use Set2 to differentiate from prior categoricals
plt.figure(figsize=(20, 20))
plt.pie(train['Soil_Type'].value_counts().values,
colors=cmap,
labels=train['Soil_Type'].value_counts().keys(),
autopct='%.1f%%')
#This gives the decimal points used
plt.title("Soil Type Distribution - Training Data")
#plt.show()

#Visualization of Soil_Type in testing set using value_counts
cmap = sns.color_palette("Set2", as_cmap=True)(np.arange(7))
#Use Set2 to differentiate from prior categoricals
plt.figure(figsize=(20, 20))
plt.pie(test['Soil_Type'].value_counts().values,
colors=cmap,
labels=test['Soil_Type'].value_counts().keys(),
autopct='%.1f%%')
#This gives the decimal points used
plt.title("Soil Type Distribution - Testing Data")
#plt.show()

"""Although not pretty for very small values, the above is informative, indicating that there are a large number of soil types that are infrequent, and several dominant soil types. Also, the training and testing sets have very similar soil types. I will be able to combine many of these smaller variables below.

### Visualize variable distribution densities
"""

dens_data = [f for f in train.columns]
f = pd.melt(train, value_vars=dens_data)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

"""The above indiactes a very nice, normal looking distribution for Hillshade_3pm in particular, with some others displaying both left and right skewness

### Correlation Matrix
"""

#Create a correlation matrix to see if variables are correlated 
corr_data = train.copy()
corr_data = pd.get_dummies(data=corr_data, columns=['Wilderness_Area'])
correlation_matrix = corr_data.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar = True, square= True, fmt='.2f', annot=True, annot_kws={'size':15}, cmap = 'coolwarm')

"""I can disregard Soil_Type as it is a categorical variable

There is a positive correlation of .65 for Hillshade_3pm and Aspect. 

As predicted, Wilderness_Area and Elevation have some correlation, particularly Wilderness_Area 4 and negative correlation of .69 with elevation. 

There is a fairly high negative correlation between Hillshade_9am and Hillshade_3pm, which makes perfect sense given the movement of the sun throughout the day. Hillshade_9am and Aspect also have a negative correlation of -.58.

# Data Preprocessing

## Turn target binary
"""

def separate_binary(train, column, target_value):
  y = train[column]
  x = train.drop([column], axis=1)
  #x = x
  y = y == target_value
  y = y.astype(int)

  return x, y
  
x_train, y = separate_binary(train, 'Cover_Type', 7)
#Did not call it y_train because there is no y_test, so more efficient for me to leave y

"""##Scale data"""

#from function defined in beginning
x_train, test = scale_min2dum(x_train, test, 'Wilderness_Area', 'Soil_Type')

"""## Deal with insignifcant dummies"""

#function that marks insignificant dummies as other
def insignificant_dummies_other(dummy_col, threshold):

    # removes the bind
    dummy_col = dummy_col.copy()

    # what is the ratio of a dummy in whole column
    count = pd.value_counts(dummy_col) / len(dummy_col)

    # cond whether the ratios are higher than the threshold
    mask = dummy_col.isin(count[count > threshold].index)

    # replace the ones which ratio is lower than the threshold by a special name
    dummy_col[~mask] = "Other"

    return pd.get_dummies(dummy_col, prefix=dummy_col.name)

train_soils = train['Soil_Type']
test_soils = test['Soil_Type']
#Glancing at pie chart above, I know that .031 threshold will make same changes to test and training set
train_soils = insignificant_dummies_other(train_soils, threshold=.031)
test_soils = insignificant_dummies_other(test_soils, threshold=.031)
#drop one to avoid dummy variable trap. This was not in the function because I don't want to drop first
train_soils = train_soils.drop(['Soil_Type_10'], axis=1)
test_soils = test_soils.drop(['Soil_Type_10'], axis=1)
#Now drop original from dataframe
x_train = x_train.drop(['Soil_Type'], axis=1)
test = test.drop(['Soil_Type'], axis=1)

#Do same for Wilderness_Area just in case more data is added down the line, though there will be no other column
train_wilderness = x_train['Wilderness_Area']
test_wilderness = test['Wilderness_Area']
train_wilderness = insignificant_dummies_other(train_wilderness, threshold=.01)
test_wilderness = insignificant_dummies_other(test_wilderness, threshold=.01)
#drop Wilderness_Area_4 to avoid dummmy variable trap. I did this manually instead of Dropfirst because
#looking at the data, Cover_Type 7 never occurs with Wilderness_Area_4, so for interpretability the other 3 will be useful
train_wilderness = train_wilderness.drop(['Wilderness_Area_4'], axis=1)
test_wilderness = test_wilderness.drop(['Wilderness_Area_4'], axis=1)
x_train = x_train.drop(['Wilderness_Area'], axis=1)
test = test.drop(['Wilderness_Area'], axis=1)

#Add soil_type and wilderness dummies back into dataframe. Could also wait to do this until after plf transform
x_train = pd.concat([x_train, train_soils], axis=1)
test = pd.concat([test, test_soils], axis=1)
x_train = pd.concat([x_train, train_wilderness], axis=1)
test = pd.concat([test, test_wilderness], axis=1)
