# @Author AzAkhtyamov
# https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48193

inputs = Input((75,75,5))
conv1 = Conv2D(64, (9, 9), padding='valid', activation='elu')(inputs)
conv1 = BatchNormalization(momentum = 0.99)(conv1)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
drop1 = Dropout(0.1)(pool1)

conv2 = Conv2D(64, (5, 5), padding='valid', activation='elu')(drop1)
conv2 = BatchNormalization(momentum = 0.95)(conv2)
pool2 = AvgPool2D(pool_size=(2, 2))(conv2)
drop2 = Dropout(0.1)(pool2)

conv3 = Conv2D(64, (3, 3), padding='valid', activation='elu')(drop2)
conv3 = BatchNormalization(momentum = 0.95)(conv3)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
drop3 = Dropout(0.1)(pool3)

conv4 = Conv2D(64, (3, 3), padding='valid', activation='elu')(drop3)
pool4 = AvgPool2D(pool_size=(2, 2))(conv4)

gp = GlobalMaxPooling2D()(pool4)

out = Dense(1, activation = 'sigmoid')(gp)