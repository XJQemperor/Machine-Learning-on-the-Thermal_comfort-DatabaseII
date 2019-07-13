#RF classifier for calculating 系数矩阵
from sklearn.model_selection import train_test_split #测试集、训练集划分
import pandas as pd  #统计分析库pandas+numpy
import numpy as np
from sklearn import preprocessing #数据预处理库
import seaborn as sns    
import matplotlib.pyplot as plt



features = pd.read_excel('分类过后的数据集 （分七类）.xlsx')  #分七类的数据
features = pd.read_excel('分类过后的数据集（分三类）.xlsx')   #分三类的数据
labels = np.array(features['TS'])    #分labels和features
feature = features.drop('TS', axis = 1)
feature_list = list(feature.columns) 


x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2)
def encode_features(x_train,x_test):
        features = ['Season','Building','Mode','Sex']
        df_combined = pd.concat([x_train[features],x_test[features]])
        for feature in features:
            le = preprocessing.LabelEncoder()
            le = le.fit(df_combined[feature])
            x_train[feature] = le.transform(x_train[feature])
            x_test[feature] = le.transform(x_test[feature])
        return x_train, x_test
x_train, x_test = encode_features(x_train,x_test)
###randomforest分类器 将max_depth和树的个数n_estimator作为变量


#获取相关系数矩阵   
cm = np.corrcoef(x_train.values.T)    
#设置字的比例    
sns.set(font_scale=1)    
#绘制相关系数图    
hm = sns.heatmap(cm,cbar=True,annot=True,fmt=".2f",annot_kws={"size":4},
                 yticklabels=feature_list,xticklabels=feature_list)    
plt.show()






    

