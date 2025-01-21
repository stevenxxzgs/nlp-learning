import pandas as pd
import numpy as np
import jieba

from keras.api.preprocessing import sequence
from keras.api.optimizers import SGD, RMSprop, Adagrad
from keras.api.models import Sequential
from keras.api.layers import Dense, Embedding, Activation
from keras.api.layers import Dropout
from keras.api.layers import LSTM, GRU

# 训练预料
neg = pd.read_excel('neg.xls', header=None, index_col=None)
pos = pd.read_excel('pos.xls', header=None, index_col=None)

pos['mark'] = 1
neg['mark'] = 0
pn = pd.concat([pos, neg], ignore_index=True)

pn[0] = pn[0].astype(str)
neglen = len(neg)
poslen = len(pos)

cw = lambda x: list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)

comment = pd.read_excel('sum.xls')
comment = comment[comment['rateContent'].notnull()]
comment['words'] = comment['rateContent'].apply(cw)

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)

w = []
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts())
del w, d2v_train
dict['id'] = list(range(1, len(dict) + 1))
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)
maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
print('pn: ', pn)
print('pn[mark]: ', pn['mark'])
pn = pn.sample(frac=1, random_state=42).reset_index(drop=True)
print('pn: ', pn)
print('pn[mark]: ', pn['mark'])
x = np.array(list(pn['sent']))[::2]
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent']))
ya = np.array(list(pn['mark']))
print("x.shape: ", x.shape)
print("y.shape: ", y.shape)
print("xt.shape: ", xt.shape)
print("yt.shape: ", yt.shape)
print("xa.shape: ", xa.shape)
print("ya.shape: ", ya.shape)
print(yt)
model = Sequential()
model.add(Embedding(len(dict)+1, 256))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=16, epochs=30)
predictions = model.predict(xt)
np.savetxt('predictions.csv', predictions, delimiter=',', fmt='%f')

# 有坑，不应该是int32，应该是int64，因为yt是int64，而且数据维度也要保持一致，不然mean算出来会错
# classes = (predictions > 0.5).astype("int32")
classes = (predictions > 0.5).astype("int64")
# 而且数据性状也要一致，不能是(y_num, 1)
classes = classes.reshape(-1)
# # 检查数据形状
np.savetxt('yt.csv', yt, delimiter=',', fmt='%d')
print("classes 的数据类型:", classes.dtype)
print("yt 的数据类型:", yt.dtype)
print("classes 的形状:", classes.shape)
print("yt 的形状:", yt.shape)

from sklearn.metrics import accuracy_score
acc = accuracy_score( yt,classes)
print("Accuracy: ", acc)
acc = np.mean(classes == yt)
print("acc: ", acc)