import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import random
from sklearn import svm
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import FileStream
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.neural_networks import PerceptronMask



def stringToList(string):
    listRes = list(string.split(","))
    return listRes

def plaintext_to_value(text):
    dic = {'email':1,
    'banking': 2,
    'healthcare': 3,
    'door locker':4,
    'camera': 5,
    'call assistant': 6,
    'video call': 7,
    'location': 8,
    'voice recording': 9,
    'todo': 10,
    'sleep hours': 11,
    'playlists': 12,
    'thermostat': 13,
    'shopping': 14,
    'weather': 15,
    'your parents':1, 
    'your partner':2, 
    'your siblings':3, 
    'your housemates':4, 
    'your children':5, 
    'neighbours':6, 
    'close friends':7,
    'your friends':7, 
    'close family':8, 
    'house helper/keeper':9,
    'house keeper':9,
    'house hepler/keeper':9, 
    'house keeper/helper':9, 
    'visitors in general':10, 
    'assistant provider':11,
    'skills':12,
    'other skills':13,
    'advertising agencies':14,
    'law enforcement agencies':15,
    'no purpose&no condition': 1,
    'with purpose&no condition':1,
    'with purpose&condition1':2,
    'with purpose&condition2':3,
    'with purpose&condition3': 4,
    'with purpose&condition4': 5,
    'with purpose&condition5': 6,
    'neutral': 0,
    'completely acceptable': 1,
    'somewhat acceptable': 1,
    'completely unacceptable': 0,
    'somewhat unacceptable': 0,
    'neutral acceptable': 1
    }
    return dic[text]


def trans_data(plain_filename):
    '''
    This function is used to process the original data file,
    and split the dataset to a training set and a test set,
    then the data could be used for k-fold validation.

    Input: plaintext-file
    Output: numerical-list
    '''

    f = open(plain_filename)
    reader = csv.reader(f)
    list_data = []
    for index, row in enumerate(reader):
        line_list = []
        for num, item in enumerate(row):
            buffer = []
            if len(item) != 0 :
                item = item.strip("][")
                new_item = stringToList(item)
                buffer.append(plaintext_to_value(new_item[0].strip("' '")))
                buffer.append(plaintext_to_value(new_item[1].strip("' '")))
                buffer.append(plaintext_to_value(new_item[2].strip("' '")))
                buffer.append(plaintext_to_value(new_item[-1].strip("' '")))
                line_list.append(buffer)
        list_data.append(line_list)
    

    return list_data




def train_test_split(data_list, i):
    train_list = []
    test_list = []
    for index, rows in enumerate(data_list):
        if index == i:
            test_list.append(rows)
        else:
            train_list.append(rows)

    test_array = np.array(test_list[0])
    label_test = []
    for x in test_list[0]:
        label_test.append(x[-1])
    test_len = len(label_test)
    # label_test.append(x[-1] for x in test_list[0])
    df_1 = pd.DataFrame(test_array, columns = ['datatype', 'recipient', 'condition', 'class'])
    test_df = pd.DataFrame(df_1, columns = ['datatype', 'recipient', 'condition'])
    test_label_df = pd.DataFrame(df_1, columns = ['class'])
    X_test = test_df.to_numpy()
    y_test = np.array(label_test)


    train_data_in_list = []
    for num, row in enumerate(train_list):
        for index, value in enumerate(row):
            train_data_in_list.append(value)

    label_train = []
    for x in train_data_in_list:
        label_train.append(x[-1])
    df_2 = pd.DataFrame(train_data_in_list, columns = ['datatype', 'recipient', 'condition', 'class'])
    train_df = pd.DataFrame(df_2, columns = ['datatype', 'recipient', 'condition'])
    train_label_df = pd.DataFrame(df_2, columns = ['class'])
    X_train = train_df.to_numpy()
    y_train = np.array(label_train)
    # import pdb; pdb.set_trace()

    return X_train, X_test, y_train, y_test, test_len



def data_stream_generator(X_train, X_test, y_train, y_test):
    
    list_train_X = X_train.tolist()
    list_test_X = X_test.tolist()
    new_X_train_list = list_train_X + list_test_X

    list_train_y = y_train.tolist()
    list_test_y = y_test.tolist()
    new_y_train_list = list_train_y + list_test_y
    if len(new_y_train_list) != len(new_X_train_list):
        raise ValueError  # shorthand for 'raise ValueError()'
    else:
        new_X_train = np.array(new_X_train_list)
        new_y_train = np.array(new_y_train_list)
    return new_X_train, new_y_train




data = "../../Data/final_data.csv"  #上上级目录中找到data file
data_list = trans_data(data)
random.seed(1)
random.shuffle(data_list)
# iteration_number = len(data_list) 
iteration_number = len(data_list)
accuracy_record = []

neural_network_analysis = 1

for i in range(0, iteration_number):
    X_train, X_test, y_train, y_test, test_len = train_test_split(data_list, i)
    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

    new_X_train, new_y_train = data_stream_generator(X_train, X_test, y_train, y_test)

    stream = DataStream(new_X_train,new_y_train, cat_features=None, name=None, allow_nan=False)
    ht = HoeffdingTreeClassifier()
    aht = HoeffdingAdaptiveTreeClassifier()
    nb = NaiveBayes()


    if neural_network_analysis == 1:
    #new method proposed:
    
        perceptron = PerceptronMask()
        neural_stream = DataStream(X_train,y_train, cat_features=None, name=None, allow_nan=False)
        n_samples = 0
        correct_cnt = 0
        perceptron.fit(X_train, y_train, classes=stream.target_values)
        user_stream = DataStream(X_test,y_test, cat_features=None, name=None, allow_nan=False)
        
        while n_samples < len(X_test) and user_stream.has_more_samples():
            X, y = user_stream.next_sample()
            my_pred = perceptron.predict(X)
            if y[0] == my_pred[0]:
                correct_cnt += 1
            perceptron.partial_fit(X, y, classes=stream.target_values)
            n_samples += 1

         # Display the results
        print('Perceptron Mask usage example')
        print('{} samples analyzed'.format(n_samples))
        print("Perceptron's performance: {}".format(correct_cnt / n_samples))
        this_neural_accuracy = correct_cnt / n_samples
        accuracy_record.append(this_neural_accuracy)
        # import pdb; pdb.set_trace()
        #End of neural networks


    else:
        #Normal method to train the model
        size = len(new_X_train) - test_len
        evaluator = EvaluatePrequential(n_wait=1,show_plot=False, pretrain_size=size,
            max_samples=len(new_X_train) + 1,
            metrics=['accuracy', 'kappa','precision','recall'])
        model_choose = aht
        evaluator.evaluate(stream=stream, model= model_choose)
        import skmultiflow
        this_user_accuracy = evaluator._data_buffer.get_data(metric_id=skmultiflow.utils.constants.ACCURACY, 
            data_id=skmultiflow.utils.constants.MEAN)[0]
        accuracy_record.append(this_user_accuracy)
        print("Method name : ", model_choose)
        # import pdb; pdb.set_trace()

print("Accuracy for this method:", sum(accuracy_record)/len(accuracy_record))
#import pdb; pdb.set_trace()









