from sklearn.ensemble import RandomForestClassifier  #引入RandomClassifier随机森林分类器
from sklearn.model_selection import train_test_split #测试集、训练集划分
import pandas as pd  #统计分析库pandas+numpy
import numpy as np
from sklearn import preprocessing #数据预处理库
import seaborn as sns    



features = pd.read_excel('分类过后的数据集 （分七类）.xlsx')  #分七类的数据
features = pd.read_excel('分类过后的数据集（分三类）.xlsx')   #分三类的数据
labels = np.array(features['TS'])    #分labels和features
features = features.drop('TS', axis = 1)
feature_list = list(features.columns) 
Min=1 #设置最大值和最小值是方便为了得到accuracy的区间
Max=0

Accuracy=[] #得到的accuracy作为列表储存


for i in range(200): 
#测试200次，在100-149中挑选出树的个数，20-39中挑出随机森林的层数
#
    
    
    
    
###分割训练集与测试集
    for j in range(100,150):
        for k in range(20,40):
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
    
    clf = RandomForestClassifier(class_weight='balanced',
                                 random_state=1,n_estimators=j,
                                 max_depth=k ,max_features='auto')
    result = clf.fit(x_train,y_train).predict(x_test) ###作为预测的结果
    Accuracy.append(clf.score(x_test,y_test)) ###列表推导
    if(clf.score(x_test,y_test)>Max):###每次循环代替最大最小值
        Max=clf.score(x_test,y_test)
###记录出现最大次数的顺序、以及accuracy达到最大时树的个数和层数  
        iMax=i;                  
        jMax=j;               
        kMax=k;
    if(clf.score(x_test,y_test)<Min):
        Min=clf.score(x_test,y_test)
        





sns.set_style('darkgrid')
sns.distplot(Accuracy)       
###查看accuracy的分布直方图，可以直观看出用随机森林训练的accuracy的概率最大的区间




print("i= ",iMax)         ###将结果输出
print("j= ",jMax)
print("k= ",kMax)
print("Max= ",Max)
print("Min= ",Min)

     
    
    
    













