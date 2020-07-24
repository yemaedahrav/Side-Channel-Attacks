import pandas as pd
import numpy as np

import datetime

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 

import matplotlib.pyplot as plt  
import seaborn as sns 

from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

room1 = pd.read_csv('data_test_room1.csv')
room2 = pd.read_csv('data_test_room2.csv')
room3 = pd.read_csv('data_test_room3.csv')

plt.plot(room1.index.values,room1.B_net,label='B1',color='blue',linewidth='1')
plt.plot(room2.index.values,room2.B_net,label='B2',color='red',linewidth='1')
plt.plot(room3.index.values,room3.B_net,label='B3',color='green',linewidth='1')
plt.xlabel('Timestamp') 
plt.tick_params(
    axis='x',         
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.ylabel('B') 
plt.title('All rooms B') 
plt.savefig('All_Rooms_B')
plt.legend()
plt.show() 

plt.plot(room1.index.values,room1.B_x,label='B1',color='blue',linewidth='1')
plt.plot(room2.index.values,room2.B_x,label='B2',color='red',linewidth='1')
plt.plot(room3.index.values,room3.B_x,label='B3',color='green',linewidth='1')
plt.xlabel('Timestamp') 
plt.tick_params(
    axis='x',         
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.ylabel('Bx') 
plt.title('All rooms Bx') 
plt.savefig('All_Rooms_Bx')
plt.legend()

plt.plot(room1.index.values,room1.B_y,label='B1',color='blue',linewidth='1')
plt.plot(room2.index.values,room2.B_y,label='B2',color='red',linewidth='1')
plt.plot(room3.index.values,room3.B_y,label='B3',color='green',linewidth='1')
plt.xlabel('Timestamp') 
plt.tick_params(
    axis='x',         
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.ylabel('By') 
plt.title('All rooms By') 
plt.savefig('All_Rooms_By')
plt.legend()

plt.plot(room1.index.values,room1.B_z,label='B1',color='blue',linewidth='1')
plt.plot(room2.index.values,room2.B_z,label='B2',color='red',linewidth='1')
plt.plot(room3.index.values,room3.B_z,label='B3',color='green',linewidth='1')
plt.xlabel('Timestamp') 
plt.tick_params(
    axis='x',         
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.ylabel('Bz') 
plt.title('All rooms Bz') 
plt.savefig('All_Rooms_Bz')
plt.legend()