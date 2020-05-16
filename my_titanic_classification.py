import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def preprocessing(data):

    # Embarked pre processing
    data['Embarked'].replace('S', 1, inplace=True)
    data['Embarked'].replace('C', 2, inplace=True)
    data['Embarked'].replace('Q', 3, inplace=True)

    # Cabin pre processing
    data['Cabin'].fillna('0', inplace=True)
    data.loc[data['Cabin'].str[0] == 'A', 'Cabin'] = 1
    data.loc[data['Cabin'].str[0] == 'B', 'Cabin'] = 2
    data.loc[data['Cabin'].str[0] == 'C', 'Cabin'] = 3
    data.loc[data['Cabin'].str[0] == 'D', 'Cabin'] = 4
    data.loc[data['Cabin'].str[0] == 'E', 'Cabin'] = 5
    data.loc[data['Cabin'].str[0] == 'F', 'Cabin'] = 6
    data.loc[data['Cabin'].str[0] == 'G', 'Cabin'] = 7
    data.loc[data['Cabin'].str[0] == 'T', 'Cabin'] = 8

    # Sex pre processing
    data['Sex'].replace('female', 1, inplace=True)
    data['Sex'].replace('male', 2, inplace=True)

    # Missing value pre processing
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].median(), inplace=True)

    return data

def group_titles(data):

    data['Names'] = data['Name'].map(lambda x: len(re.split(' ', x)))
    data['Title'] = data['Name'].map(lambda x: re.search(', (.+?) ', x).group(1))
    data['Title'].replace('Master.', 0, inplace=True)
    data['Title'].replace('Mr.', 1, inplace=True)
    data['Title'].replace(['Ms.', 'Mlle.', 'Miss.'], 2, inplace=True)
    data['Title'].replace(['Mme.', 'Mrs.'], 3, inplace=True)
    data['Title'].replace(['Dona.', 'Lady.', 'the Countess.',
                           'Capt.', 'Col.', 'Don.', 'Dr.', 'Major.',
                           'Rev.', 'Sir.', 'Jonkheer.', 'the'], 4, inplace=True)

    return data

def data_subset(data):
    features = features = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Names',
                           'Title', 'Age', 'Cabin', 'Fare', 'Embarked']
    length_features = len(features)
    subset = data[features]
    return subset, length_features

def create_model(train_set_size, input_length, num_epochs, batch_size):

    model = Sequential()
    model.add(Dense(7, input_dim=input_length, activation='softplus'))
    model.add(Dense(3, activation='softplus'))
    model.add(Dense(1, activation='softplus'))

    my_optimizer = Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy', optimizer=my_optimizer, metrics=['accuracy'])
    weights_filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(weights_filepath, monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callback_list = [checkpoint]
    print("Call Back List: ", callback_list)

    history_model = model.fit(x_train[:train_set_size], y_train[:train_set_size], callbacks=callback_list,
                              epochs=num_epochs, batch_size=batch_size, verbose=0)

    return model, history_model

def plots(history):

    loss_history = history.history['loss']
    acc_history = history.history['accuracy']
    epochs = [(i+1) for i in range(num_epochs)]

    ax = plt.subplot(211)
    ax.plot(epochs, loss_history, color='red')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Epoch")

    ax2 = plt.subplot(212)
    ax2.plot(epochs, acc_history, color='blue')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy per Epoch")

    plt.subplots_adjust(hspace=0.8)
    plt.savefig("Output.png")
    plt.show()

def test(batch_size):
    test = pd.read_csv("test.csv", header=0)
    test_ids = test['PassengerId']

    test = preprocessing(test)
    group_titles(test)
    testdata, _ = data_subset(test)

    x_test = np.array(testdata).astype(float)

    output = model.predict(x_test, batch_size=batch_size, verbose=0)
    output = output.reshape((418,))

    outputBin = np.zeros(0)
    for element in output:
        if element <= 0.5:
            outputBin = np.append(outputBin, 0)
        else:
            outputBin = np.append(outputBin, 1)

    output = np.array(outputBin).astype(int)

    column_1 = np.concatenate((['PassengerId'], test_ids), axis=0)
    column_2 = np.concatenate((['Survived'], output), axis=0)

    file = open("output.csv", 'w')
    writer = csv.writer(file)
    for i in range(len(column_2)):
        writer.writerow([column_1[i]] + [column_2[i]])
    file.close()


np.random.seed(7)

train = pd.read_csv("train.csv")

preprocessing(train)
group_titles(train)

num_epochs = 100
batch_size = 30


traindata, length_features = data_subset(train)

y_train = np.array(train['Survived']).astype(int)
x_train = np.array(traindata).astype(float)

train_set_size = int(0.67 * len(x_train))

model, history_model = create_model(train_set_size, length_features, num_epochs, batch_size)

plots(history_model)

x_validation = x_train[train_set_size:]
y_validation = y_train[train_set_size:]

loss_and_metrics = model.evaluate(x_validation, y_validation, batch_size=batch_size)
test(batch_size)