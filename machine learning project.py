import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/10.%20RNN%20Text%20Classification/dataset/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin1')
print('총 샘플의 수 :',len(data))

data[:10]

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data[:10]

data.info()

print('결측값 여부 :',data.isnull().values.any())   #NULL값이 있나 확인

print('v2열의 유니크한 값 :',data['v2'].nunique())    #중복 값이 있나 확인, 5572개의 샘플 수중 5169의 유니크한 값 존재

data.drop_duplicates(subset=['v2'], inplace=True)   #중복값을 하나만 남기고 제거
print('총 샘플의 수 :',len(data))

data['v1'].value_counts().plot(kind='bar')

print('정상 메일과 스팸 메일의 개수')
print(data.groupby('v1').size().reset_index(name='count'))

print(f'정상 메일의 비율 = {round(data["v1"].value_counts()[0]/len(data) * 100,3)}%')
print(f'스팸 메일의 비율 = {round(data["v1"].value_counts()[1]/len(data) * 100,3)}%')

X_data = data['v2']
y_data = data['v1']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

print('--------훈련 데이터의 비율-----------')
print(f'정상 메일 = {round(y_train.value_counts()[0]/len(y_train) * 100,3)}%')
print(f'스팸 메일 = {round(y_train.value_counts()[1]/len(y_train) * 100,3)}%')

print('--------테스트 데이터의 비율-----------')
print(f'정상 메일 = {round(y_test.value_counts()[0]/len(y_test) * 100,3)}%')
print(f'스팸 메일 = {round(y_test.value_counts()[1]/len(y_test) * 100,3)}%')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train)   #Tokenizer를 사용해서 one-hot encoding수행
for i in range(5):        # 다섯개의 열 출력
    print(X_train_encoded[i])


print("X_train_encoded의 수:",format(len(X_train_encoded)))

word_to_index = tokenizer.word_index
print(word_to_index)    # 단어가 매우 많기에 생략
print("총 단어의 수: ", format(len(word_to_index)))

threshold = 2
total_cnt = len(word_to_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))    #패딩을 위한 토큰인 0번 단어를 포함시킨다.  패딩이란 데이터에 특정값을 태워서 데이터의 크기를 조정하는 것

print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded)) # 데이터로 가지고 있는 메일 중 가장 긴 메일은 189개의 단어를 가지고 있다.
print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))
plt.hist([len(sample) for sample in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = 189
X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len) # 0을 추가하여메일들의 길이를 가장 긴 메일과 같게 만든다. 패딩
X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)
print("훈련 데이터의 크기(shape):", X_train_padded.shape)

from sklearn.linear_model import SGDClassifier  #확률적 경사 하강법 분류 모델
from sklearn.metrics import accuracy_score

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train_padded, y_train)

sgd_pred = sgd_clf.predict(X_test_padded)
print('예측정확도:{0:.2f}'.format(accuracy_score(y_test,sgd_pred)))

from sklearn.model_selection import cross_val_score #분류모델은 정확도가 높다고 좋은게 아니다.

cross_val_score(sgd_clf, X_train_padded, y_train, cv=4, scoring="accuracy") #교차검증 결과 정확도가 일정하지 않고 편차가 심한것을 알 수 있다. 

from sklearn.metrics import confusion_matrix  #성능측정 오차행렬

y_train_pre_no_cv = sgd_clf.predict(X_train_padded)
confusion_matrix(y_train, y_train_pre_no_cv)

from sklearn.model_selection import cross_val_predict #오차행렬, 정확도(accuracy)

y_train_pred = cross_val_predict(sgd_clf, X_train_padded, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)

y_scores = cross_val_predict(sgd_clf, X_train_padded, y_train, cv=3, method="decision_function")

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)
plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") # Not shown
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  # Not shown
plt.plot([4.837e-3], [0.4368], "ro")               # Not shown
save_fig("roc_curve_plot")                         # Not shown
plt.show()

from sklearn.metrics import roc_auc_score #AUC 값은 클수록 좋다.

roc_auc_score(y_train, y_scores)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier   #DecisionTreeClassifier 사용
#from sklearn.model_selection import train_test_split

model_dtree = DecisionTreeClassifier(max_depth=3, random_state=20)
model_dtree.fit(X_train_padded, y_train)

#from sklearn.metrics import accuracy_score
pred = model_dtree.predict(X_test_padded)
print('예측정확도:{0:.2f}'.format(accuracy_score(y_test,pred)))

cross_val_score(model_dtree, X_train_padded, y_train, cv=4, scoring="accuracy")

from sklearn.metrics import confusion_matrix  #성능측정 오차행렬

y_train_pre_no_cv = model_dtree.predict(X_train_padded)
confusion_matrix(y_train, y_train_pre_no_cv)

from sklearn.model_selection import cross_val_predict #오차행렬, 정확도(accuracy)

y_train_pred = cross_val_predict(model_dtree, X_train_padded, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)

y_train_pred_dummy = cross_val_predict(model_dtree, X_train_padded, y_train) 
confusion_matrix(y_train, y_train_pred_dummy)

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense #RNN 모델, 마지막 시점에서 두개의 선택지 중 하나를 선택하는 이진 분류 모델
from tensorflow.keras.models import Sequential

embedding_dim = 32
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train_padded, y_train, epochs=4, batch_size=64, validation_split=0.2)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test_padded, y_test)[1]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

