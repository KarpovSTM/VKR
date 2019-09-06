import warnings
import itertools
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score,roc_curve, accuracy_score, precision_score, roc_auc_score, f1_score,confusion_matrix
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

def custom_confusion_matrix(cm, classes, normalize=False,title=''):
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title(title, fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')

def custom_roc_curve(FPR, TPR, roc_name):
    plt.figure(figsize=(5,5))
    plt.title(roc_name, fontsize=16)
    plt.plot(FPR, TPR, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')# диагональ
    plt.xlabel('FPR метрика', fontsize=16)
    plt.ylabel('TPR метрика', fontsize=16)
    plt.axis([-0.01,1,0,1])
    plt.show()




# Фильтр предупреждений
warnings.simplefilter("ignore")

original_df = pd.read_csv("C:/Users/karpo/Desktop/VKR/CreditCardOne/DiplomDataset.csv")

print(original_df.head)

print('Честные транзакции', round(original_df['Class'].value_counts()[0]/len(original_df) * 100,2), '% от всех операций')
print('Мошеннические транзакции', round(original_df['Class'].value_counts()[1]/len(original_df) * 100,2), '% от всех операций')

colors = ["#FFA500", "#20B2AA"]

sns.countplot('Class', data=original_df, palette=colors)
plt.title('Класс транзакции', fontsize=16)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,5))

ax1.hist(original_df.Time[original_df.Class == 1], bins = 100)
ax1.set_title('Мошеннические транзакции')
ax2.hist(original_df.Time[original_df.Class == 0], bins = 100)
ax2.set_title('Честные транзакции')

plt.xlabel('Время (сек.)')
plt.ylabel('Число транзакций')
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,5))
ax1.hist(original_df.Amount[original_df.Class == 1], bins = 5)
ax1.set_title('Мошеннические транзакции')
ax2.hist(original_df.Amount[original_df.Class == 0], bins = 5)
ax2.set_title('Честные транзакции')
plt.xlabel('Сумма')
plt.ylabel('Число транзакций')
plt.yscale('log')
plt.show()

# Масштабирование
rob_scaler = RobustScaler()

original_df['rubust_time'] = RobustScaler().fit_transform(original_df['Time'].values.reshape(-1,1))
original_df['robust_amount'] = RobustScaler().fit_transform(original_df['Amount'].values.reshape(-1,1))

original_df.drop(['Time','Amount'], axis=1, inplace=True)

rubust_time = original_df['rubust_time']
robust_amount = original_df['robust_amount']

original_df.drop(['robust_amount', 'rubust_time'], axis=1, inplace=True)

original_df.insert(0, 'robust_amount', robust_amount)
original_df.insert(1, 'rubust_time', rubust_time)

original_df.head

# Случайное разбиение оригинальных данных на обучающие и тестовые выборки
X = original_df.drop('Class', axis=1)# Данные без целевого признака
y = original_df['Class']# Целевой признак

strat_cross_val = StratifiedShuffleSplit(n_splits=10,test_size= 0.2, random_state=123)

for index_train, index_test in strat_cross_val.split(X, y): 
    original_train_data, original_test_data = X.iloc[index_train], X.iloc[index_test]
    original_train_index, original_test_index = y.iloc[index_train], y.iloc[index_test]
    
# Переведем в массив
original_train_data = original_train_data.values
original_test_data = original_test_data.values
original_train_index = original_train_index.values
original_test_index = original_test_index.values


    
rf_prop = {'criterion': ['entropy', 'gini'], 'max_depth': [10, 20, 100], 'n_estimators': [10, 20, 100]}
svc_prop = { 'C': [0.4, 0.5, 0.6, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'] }
knn_prop = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 8, 9, 10] , 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
lr_prop = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Реализация функции тестирования классификаторов 
def ClassifierTesting(method_name,method, method_prop, sampling_algoritm):
    
    acc_lst = []
    prec_lst = []
    rec_lst = []
    f1_lst = []
    auc_lst = []
    
    random_search = RandomizedSearchCV(method, method_prop, n_iter=20)
    
    for train, test in strat_cross_val.split(original_train_data, original_train_index):

        cross_val_model = imbalanced_make_pipeline(sampling_algoritm, random_search) # SMOTE happens during Cross Validation not before..
        cross_val_model.fit(original_train_data[train], original_train_index[train])
        
        best_est = random_search.best_estimator_
        prediction = best_est.predict(original_train_data[test])
        
        acc_lst.append(cross_val_model.score(original_train_data[test], original_train_index[test]))
        prec_lst.append(precision_score(original_train_index[test], prediction)) 
        rec_lst.append(recall_score(original_train_index[test], prediction))
        f1_lst.append(f1_score(original_train_index[test], prediction))
        auc_lst.append(roc_auc_score(original_train_index[test], prediction))

    print('_' * 50)
    print(method_name)   
    print('_' * 50)
    print('Результат перекрестной проверки:')
    print("accuracy:{0:.3f}".format(np.mean(acc_lst)*100),'%')
    print("precision:{0:.3f}".format(np.mean(prec_lst)*100),'%')
    print("recall:{0:.3f}".format(np.mean(rec_lst)*100),'%')
    print("f1:{0:.3f}".format(np.mean(f1_lst)*100),'%')
    print("Roc Auc:{0:.3f}".format(np.mean(auc_lst)*100),'%')
    print('_' * 50)
    print('_' * 50)
    
    acc_lst = []
    prec_lst = []
    rec_lst = []
    f1_lst = []
    auc_lst = []
        
    prediction = best_est.predict(original_test_data)
    
    acc_lst.append(accuracy_score(original_test_index, prediction))
    prec_lst.append(precision_score(original_test_index, prediction)) 
    rec_lst.append(recall_score(original_test_index, prediction))
    f1_lst.append(f1_score(original_test_index, prediction))
    auc_lst.append(roc_auc_score(original_test_index, prediction))
    
    print('Результат тестирования:')    
    print("accuracy:{0:.3f}".format(np.mean(acc_lst)*100),'%')
    print("precision:{0:.3f}".format(np.mean(prec_lst)*100),'%')
    print("recall:{0:.3f}".format(np.mean(rec_lst)*100),'%')
    print("f1:{0:.3f}".format(np.mean(f1_lst)*100),'%')
    print("Roc Auc:{0:.3f}".format(np.mean(auc_lst)*100),'%')
    print('_' * 50)
    
    cm = confusion_matrix(original_test_index, prediction)
    
    fig, ax = plt.subplots(1, 1,figsize=(5,5))
    sns.heatmap(cm, ax=ax, annot=True, cmap=plt.cm.Purples)
    ax.set_title(method_name, fontsize=18)
    ax.set_xticklabels(['Честные', 'Мошеннические'], fontsize=10, rotation=0)
    ax.set_yticklabels(['Честные', 'Мошеннические'], fontsize=10, rotation=90)
    ax.set_xlabel(('Предсказанные значения'), fontsize=12, rotation=0)
    ax.set_ylabel(('Истинные значения'), fontsize=12, rotation=90)
    plt.show()
    
    FPR, TPR, none = roc_curve(original_test_index, prediction)
    custom_roc_curve(FPR, TPR, method_name)
    
# Реализация функции нейронной сети
def NEURO(sampling_name,sampling_method):
    sm = sampling_method
    resample_date, resample_index = sm.fit_sample(original_train_data, original_train_index)
     
    n_inputs = resample_date.shape[1]
    neuro_model = Sequential()
    neuro_model.add(Dense(n_inputs, input_shape=(n_inputs, ), activation='relu')) 
    neuro_model.add(BatchNormalization())
    neuro_model.add(Dense(32, activation='relu'))
    neuro_model.add(BatchNormalization())
    neuro_model.add(Dense(16, activation='relu'))
    neuro_model.add(BatchNormalization())
    neuro_model.add(Dense(2, activation='sigmoid'))
    
    neuro_model.compile(Adam(lr=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    neuro_model_history=neuro_model.fit(resample_date, resample_index, validation_split=0.2, batch_size=100, epochs=100, shuffle=True,  verbose=0)
    neuro_model_predictions =neuro_model.predict_classes(original_test_data, batch_size=100, verbose=0)
    
    loss, acc = neuro_model.evaluate(original_test_data, original_test_index)
    
    print('Функция потерь:', loss*100,'%')
    print('Точность (accuracy):', acc*100,'%')
    
    neuro_model_cmr = confusion_matrix(original_test_index, neuro_model_predictions)
    
    labels = ['Честн.', 'Мошен.']
    fig = plt.figure(figsize=(10,5))
    custom_confusion_matrix(neuro_model_cmr, labels, title=sampling_name)
   
    # Вывод динамики изменения ошибки и точности.
    plt.figure()
    plt.plot(neuro_model_history.history['loss'])
    plt.plot(neuro_model_history.history['val_loss'])
    plt.title('Функция потерь')
    plt.ylabel('Значение функции потерь')
    plt.xlabel('Количество эпох')
    plt.legend(['Тренировка', 'Тест'], loc='best')
    plt.show()
    
    plt.figure()
    plt.plot(neuro_model_history.history['acc'])
    plt.plot(neuro_model_history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('Значение accuracy')
    plt.xlabel('Количество эпох')
    plt.legend(['Тренировка', 'Тест'], loc='best')
    plt.show()
"""
# Алгоритм RandomUnderSampler.
ClassifierTesting('Метод опорных векторов',SVC(), svc_prop, RandomUnderSampler(sampling_strategy='majority'))
ClassifierTesting('Случайный лес',RandomForestClassifier(), rf_prop, RandomUnderSampler(sampling_strategy='majority'))
ClassifierTesting('K-ближайших соседей',KNeighborsClassifier(), knn_prop, RandomUnderSampler(sampling_strategy='majority'))
ClassifierTesting('Логистическая регрессия',LogisticRegression(), lr_prop, RandomUnderSampler(sampling_strategy='majority'))    

# Алгоритм NearMiss.
ClassifierTesting('Метод опорных векторов',SVC(), svc_prop, NearMiss(version=1,sampling_strategy='majority',n_jobs=-1))
ClassifierTesting('Случайный лес',RandomForestClassifier(), rf_prop, NearMiss(version=1,sampling_strategy='majority',n_jobs=-1))
ClassifierTesting('K-ближайших соседей',KNeighborsClassifier(), knn_prop, NearMiss(version=1,sampling_strategy='majority',n_jobs=-1))
ClassifierTesting('Логистическая регрессия',LogisticRegression(), lr_prop, NearMiss(version=1,sampling_strategy='majority',n_jobs=-1))

# Алгоритм SMOTE.
ClassifierTesting('Метод опорных векторов',SVC(), svc_prop, SMOTE(sampling_strategy='minority',n_jobs=-1))
ClassifierTesting('Случайный лес',RandomForestClassifier(), rf_prop, SMOTE(sampling_strategy='minority',n_jobs=-1))
ClassifierTesting('K-ближайших соседей',KNeighborsClassifier(), knn_prop, SMOTE(sampling_strategy='minority',n_jobs=-1))
ClassifierTesting('Логистическая регрессия',LogisticRegression(), lr_prop, SMOTE(sampling_strategy='minority',n_jobs=-1))

# Алгоритм ADASYN.
ClassifierTesting('Метод опорных векторов',SVC(), svc_prop, ADASYN(sampling_strategy='minority',n_jobs=-1))
ClassifierTesting('Случайный лес',RandomForestClassifier(), rf_prop, ADASYN(sampling_strategy='minority',n_jobs=-1))
ClassifierTesting('K-ближайших соседей',KNeighborsClassifier(), knn_prop, ADASYN(sampling_strategy='minority',n_jobs=-1))
ClassifierTesting('Логистическая регрессия',LogisticRegression(), lr_prop, ADASYN(sampling_strategy='minority',n_jobs=-1))
"""

# Тестирование нейронной сети.
NEURO('RandomUnderSampler',RandomUnderSampler(sampling_strategy='majority',random_state=36))
NEURO('NearMiss', NearMiss(sampling_strategy='majority', version =1, random_state=36,n_jobs=-1))
NEURO('SMOTE', SMOTE(sampling_strategy='minority', random_state=36,n_jobs=-1))
NEURO('ADASYN', ADASYN(sampling_strategy='minority', random_state=36,n_jobs=-1))








