# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:04:24 2020

@author: dafen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score
import os

raw_data = pd.read_excel('alldatafrequency=2.xlsx')
data = raw_data.iloc[:,1:5].values
data = np.array(data)

# fine the resonalbe clustering
Silhouse = []
Calinski_harabasz = []
davies_bouldin = []
for i in range(2,10,1):
    ward = AgglomerativeClustering(n_clusters = i, linkage = 'ward')
    ward_1 = ward.fit(data)
    cluster_labels = ward_1.labels_
    Silhouse.append(silhouette_score(data, cluster_labels))  
    Calinski_harabasz.append(calinski_harabaz_score(data, cluster_labels))
    davies_bouldin.append(davies_bouldin_score(data, cluster_labels))
   
Silhouse = []
Calinski_harabasz = []
davies_bouldin = []

for i in range(2,10,1):
    k_means = KMeans(n_clusters = i, init = "k-means++", max_iter = 30, n_init = 10, random_state = 10)
    labels = k_means.fit_predict(data)
    Silhouse.append(silhouette_score(data, labels))  
    Calinski_harabasz.append(calinski_harabaz_score(data, labels))
    davies_bouldin.append(davies_bouldin_score(data, labels))
    
    
#hierarchical visualization
import sys
sys.setrecursionlimit(1000000)
    
import scipy.cluster.hierarchy as shc
plt.figure(figsize = (10,7))
plt.title('Customer Dendograms')
dendrogram = shc.dendrogram(shc.linkage(data, method = 'ward'))
plt.xlabel('Customer')
plt.ylabel('Euclidean distances')
plt.show()


#3d hierachy figures
ward = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')
ward_1 = ward.fit(data)
cluster_labels = ward_1.labels_
raw_data['Clustering'] = cluster_labels

raw_data.to_csv(r'RFMI-Hierarchy clustering.csv')

# K-means


k_means = KMeans(n_clusters = 7, init = "k-means++", max_iter = 30, n_init = 10, random_state = 10)
labels = k_means.fit_predict(data)
raw_data['Clustering'] = labels
raw_data.to_csv(r'RFMI-k-MEANS-7clustering.csv')

#3D visulization
ax=plt.subplot(projection='3d')
ax.scatter(raw_data[raw_data['Clustering'] ==0].iloc[:,8], raw_data[raw_data['Clustering'] ==0].iloc[:,9],raw_data[raw_data['Clustering'] ==0].iloc[:,10], c= 'red')
ax.view_init(elev=45,azim=60)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_zlabel("Monetary")
ax.set_ylabel("Frequency")
ax.set_xlabel("Recency")

ax=plt.subplot(projection='3d')
ax.scatter(raw_data[raw_data['Clustering'] ==1].iloc[:,8], raw_data[raw_data['Clustering'] ==1].iloc[:,9],raw_data[raw_data['Clustering'] ==1].iloc[:,10], c= 'blue')
ax.view_init(elev=45,azim=60)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_zlabel("Monetary")
ax.set_ylabel("Frequency")
ax.set_xlabel("Recency")

ax=plt.subplot(projection='3d')
ax.scatter(raw_data[raw_data['Clustering'] ==2].iloc[:,8], raw_data[raw_data['Clustering'] ==2].iloc[:,9],raw_data[raw_data['Clustering'] ==2].iloc[:,10], c= 'green')
ax.view_init(elev=45,azim=60)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_zlabel("Monetary")
ax.set_ylabel("Frequency")
ax.set_xlabel("Recency")

ax=plt.subplot(projection='3d')
ax.scatter(raw_data[raw_data['Clustering'] ==3].iloc[:,8], raw_data[raw_data['Clustering'] ==3].iloc[:,9],raw_data[raw_data['Clustering'] ==3].iloc[:,10], c= 'pink')
ax.view_init(elev=45,azim=60)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_zlabel("Monetary")
ax.set_ylabel("Frequency")
ax.set_xlabel("Recency")


ax=plt.subplot(projection='3d')
ax.scatter(raw_data[raw_data['Clustering'] ==4].iloc[:,8], raw_data[raw_data['Clustering'] ==4].iloc[:,9],raw_data[raw_data['Clustering'] ==4].iloc[:,10], c= 'black')
ax.view_init(elev=45,azim=60)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_zlabel("Monetary")
ax.set_ylabel("Frequency")
ax.set_xlabel("Recency")

ax=plt.subplot(projection='3d')
ax.scatter(raw_data[raw_data['Clustering'] ==5].iloc[:,8], raw_data[raw_data['Clustering'] ==5].iloc[:,9],raw_data[raw_data['Clustering'] ==5].iloc[:,10], c= 'orange')
ax.view_init(elev=45,azim=60)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_zlabel("Monetary")
ax.set_ylabel("Frequency")
ax.set_xlabel("Recency")


ax=plt.subplot(projection='3d')
ax.scatter(raw_data[raw_data['Clustering'] ==6].iloc[:,8], raw_data[raw_data['Clustering'] ==6].iloc[:,9],raw_data[raw_data['Clustering'] ==6].iloc[:,10], c= 'purple')
ax.view_init(elev=45,azim=60)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_zlabel("Monetary")
ax.set_ylabel("Frequency")
ax.set_xlabel("Recency")