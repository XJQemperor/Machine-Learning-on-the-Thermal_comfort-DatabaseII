from sklearn.ensemble import RandomForestRegressor    #Random的库
from sklearn.preprocessing import StandardScaler  #归一化的库
from sklearn.model_selection import train_test_split  #训练集和数据集划分的库
from sklearn.ensemble.partial_dependence import plot_partial_dependence #partial dependence 的包
import numpy as np   #numpy和pandas作为数据处理的常规包
import pandas as pd      
from sklearn import preprocessing   #数据预处理包
import csv #csv文件用这个，不然就是pandas库里调用read_excel读取excel文件
import matplotlib.pyplot as plt  #作图常用的pyplot库

features = pd.read_excel('Database-II chosen4.xlsx') 
features2 = pd.read_excel('归一化过后的训练集+验证集.xlsx')

print(features2.corr())   #打印相关系数矩阵，但是首先需要进行对文字变量进行labelEncode操作



labels = np.array(features['TS'])   #找出要预测的变量放到第一列，别的作为feature
features = features.drop('TS', axis = 1)
feature_list = list(features.columns)  #其实就是方便看所有的变量 

#feature全部变成数值连续型变量

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=50)
def encode_features(x_train,x_test):
    features = ['Season','Year','Climate','Country','Building','Mode','Sex']
    df_combined = pd.concat([x_train[features],x_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        x_train[feature] = le.transform(x_train[feature])
        x_test[feature] = le.transform(x_test[feature])
    return x_train, x_test

x_train, x_test = encode_features(x_train,x_test)

#80%训练集，%20验证集
#这里的自己建的函数是基于数据预处理包中的LabelEncoder,将标签变量变成数字变量
#注意不能变为独热编码是因为我们需要做一个回归器而不是一个分类器



ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test  = ss_x.fit_transform(x_test)
#x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test  = ss_y.fit_transform(y_test.reshape(-1, 1))
#y_test = ss_y.transform(y_test.reshape(-1, 1))







#利用k折交叉验证法
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
x_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(x_train, num_folds)

y_train_folds = np.array_split(y_train, num_folds)
k_to_accuracies = {}
for k in k_choices: 
    A = []    
    for i in range(num_folds):     
        x_val_k = x_train_folds[i]
        #validation set        
        y_val_k = y_train_folds[i] 
        #validation set               
        x_train_k = np.concatenate(x_train_folds[:i] + x_train_folds[i+1:])        
        y_train_k = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])            
        classifier = RandomForestRegressor()        
        classifier.train(x_train_k, y_train_k)        
        dists = classifier.compute_distances_no_loops(x_val_k)        
        y_val_pred = classifier.predict_labels(dists, k=k)               
        num_correct = np.sum(y_val_pred == y_val_k)        
        num_val = x_val_k.shape[0]        
        accuracy = float(num_correct) / num_val        
        A.append(accuracy)    
        k_to_accuracies[k] = A        
for k in sorted(k_to_accuracies):
   for accuracy in k_to_accuracies[k]:
      print('k = %d, accuracy = %f' % (k, accuracy))

#此处代码需要改进，因为会发现超出array的范围。


#采用RandomForestRegressor作回归
rfr = RandomForestRegressor(n_estimators= 1000, random_state=42) 
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_test)
predictions =rfr.predict(x_test) 
 


# 得到变量的重要性
importances = list(rfr.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)] 
#对重要性作排序并输出，主要是为了方便分析。
feature_importances = sorted(feature_importances, key = lambda x: x[1],reverse = True) 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


#做一个表并输出比较直观
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical') 
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances'); 











