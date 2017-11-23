from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

df_raw = pd.read_csv('~/Documents/2017/model1/fraud/prod_clean/data/prepared_data_snapshot.csv')
df_raw.fillna(-1, inplace=True)

target, pk = 'isfraud', 'applicationid'
oos, l = .8, len(df_raw)

# df_raw_copy = df_raw.loc[:, df_raw.columns != target]
# df_raw_copy.loc[:, df_raw_copy.columns != pk].apply(lambda x: x / 10)

# df_raw = df_raw.merge(df_raw_copy, on=pk)
# df_raw = df_raw.merge(df_raw_copy2, on=pk)

# .loc[:, tr.columns != target]

data = df_raw.loc[:, df_raw.columns != target]
y = df_raw.loc[:, target]
data = (data - data.mean()) / (data.max() - data.min())

# tr = df_raw.iloc[:round(l * oos), ]
# ts = df_raw.iloc[round(l * oos):, ]

train, labels = data.iloc[:round(l * oos),], y.iloc[:round(l * oos), ]
test, target_test = data.iloc[round(l * oos):,], y.iloc[round(l * oos):,]
# id_test = ts.ix[:, pk]

train, test = train.drop([pk], axis=1), test.drop([pk], axis=1)

X_train = (train.values).astype('float32')
X_test = (test.values).astype('float32')

# print(X_train)

# convert list of labels to binary class matrix
# y_train = np_utils.to_categorical(labels)
y_train = labels.values

# print(tr.shape, ts.shape)

# pre-processing: divide by max and substract mean
# scale = np.max(X_train)
# X_train /= scale
# X_test /= scale

# mean = np.std(X_train)

# X_train -= mean
# X_test -= mean

input_dim = X_train.shape[1]
# Uncomment for multiclass
# nb_classes = y_train.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(1000, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1, activation='sigmoid'))

# Convolutional NN. Draft
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', \
#                  input_shape=...))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# from keras.models import Sequential
# from keras.layers import Dense, Activation, Conv2D

# input_shape = 100

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01),
#               metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
#           callbacks=[history])


# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])



# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
# , metrics=['binary_accuracy']

# print("Training...")
model.fit(X_train, labels.values, epochs=2, batch_size=64, validation_split=0.15, verbose=2)

# print("Generating test predictions...")
preds = model.predict_proba(X_test, verbose=False)
# score = model.evaluate(X_test, target_test, batch_size=64)

# print(score, preds)
print(roc_auc_score(target_test, preds[:, 0]))


# def write_preds(preds, fname):
#     pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

# write_preds(preds, "keras-mlp.csv")

# 0.58445150386
# ~0.63