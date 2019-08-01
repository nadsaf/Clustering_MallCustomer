import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid", {'font.family':'serif'})
plt.style.use("seaborn-darkgrid")

#----------------------------------------------------------------------------------------------
# DataFrame

df = pd.read_csv('6.Mall_Customers.csv')
df =  df.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Score'})
# print(df)

# fig = plt.figure(figsize=(20,8))
# sns.pairplot(df, hue='Gender')
# plt.show()
# plt.close()

#----------------------------------------------------------------------------------------------
# Preprocessing & Labelling
dfdum = pd.get_dummies(df['Gender']) # Female 0 Male 1
# print(dfdum)
df = df.drop(['Gender'], axis=1)
df = pd.concat((df, dfdum), axis=1)
# print(df.isnull().sum()/len(df))

#================================================= Machine Learning ==============================================
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

# # sse = []
# # for i in range(1,15):
# #     model = KMeans(n_clusters= i)
# #     model.fit(df[['Income', 'Age', 'Score', 'Female', 'Male']])
# #     sse.append(model.inertia_)

# # print(sse)

# # # plot = elbow method
# # import matplotlib.pyplot as plt
# # plt.plot(
# #     range(1,15), 
# #     sse,
# #     marker='.'
# #     )
# # plt.xticks(range(1,15))
# # plt.grid(True)
# # plt.show()
#----------------------------------------------------------------------------------------------
model1 = KMeans(n_clusters=5, random_state=0)
model2 = KMeans(n_clusters=5, random_state=0)
# Income - Score
model1.fit(df[['Income', 'Score']])
# print(model.labels_)
df['Label'] = model1.labels_
map_income = {
    0 : 'High Income-Low Spending', 
    1 : 'Medium',
    2 : 'High Priority',
    3 : 'Low Income-High Spending',
    4 : 'Low Income-Low Spending'
    }
df['Label'] = df['Label'].map(map_income)
# print(df['Label'].tolist())
# print(model1.cluster_centers_)


# Clustering Map
x1_max = df['Income'].max() + 1
x1_min = df['Income'].min() - 1
y1_max = df['Score'].max() + 1
y1_min = df['Score'].min() - 1

X1, Y1 = np.meshgrid(
    np.arange(x1_min, x1_max, 0.01),
    np.arange(y1_min, y1_max, 0.01)
)

Z1 = model1.predict(np.c_[X1.ravel(), Y1.ravel()])
Z1 = Z1.reshape(X1.shape)

# Age - Score
model2.fit(df[['Age', 'Score']])
# print(model2.labels_)
df['Label2'] = model2.labels_
map_age = {
    0 : 'Young - Medium Spending', 
    1 : 'Young - High Spending',
    2 : 'Oldies - Medium Spending',
    3 : 'All - Low Spending',
    4 : 'Mid - Medium Spending'
    }
df['Label2'] = df['Label2'].map(map_age)
# print(df['Label2'].tolist())

# print(model2.cluster_centers_)
x2_max = df['Age'].max() + 1        
x2_min = df['Age'].min() - 1
y2_max = df['Score'].max() + 1
y2_min = df['Score'].min() - 1

X2, Y2 = np.meshgrid(
    np.arange(x2_min, x2_max, 0.01),
    np.arange(y2_min, y2_max, 0.01)
)

Z2 = model2.predict(np.c_[X2.ravel(), Y2.ravel()])
Z2 = Z2.reshape(X2.shape)

figure2= plt.figure(figsize=(12, 8))
ax = figure2.add_subplot(121)
# ax.contourf(X1,Y1,Z1, cmap= 'Blues', alpha = 0.4)
plt.imshow(Z1 , interpolation='nearest', 
           extent=(X1.min(), X1.max(), Y1.min(), Y1.max()),
           cmap = 'Blues', aspect = 'auto', alpha=0.4, origin='lower')
sns.scatterplot(df['Income'], df['Score'], hue=df['Label'], 
                palette=sns.color_palette('nipy_spectral', 5), ax=ax)
sns.scatterplot(
    model1.cluster_centers_[:,0],
    model1.cluster_centers_[:,1],
    ax=ax,
    color = 'gold',
    marker='*', s=250)
plt.title('Income vs Score')


ax1 = figure2.add_subplot(122)
# ax1.contourf(X2,Y2,Z2, cmap= 'Blues', alpha = 0.4)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(X2.min(), X2.max(), Y2.min(), Y2.max()),
           cmap = 'Blues', aspect = 'auto', alpha=0.4, origin='lower')
sns.scatterplot(df['Age'], df['Score'], hue=df['Label2'], 
                palette=sns.color_palette('nipy_spectral', 5), ax=ax1)
sns.scatterplot(
    model2.cluster_centers_[:,0],
    model2.cluster_centers_[:,1],
    ax=ax1,
    color = 'gold',
    marker='*', s=250)
plt.title('Age vs Score')
plt.show()

