import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

from sklearn import svm




iris = load_iris()

# ris.data.shape, iris.target.shape
# ((150, 4), (150,))

X_train, X_test, y_train, y_test = train_test_split(
         iris.data, iris.target, test_size=.4, random_state=0)
   #这里是按照6:4对训练集测试集进行划分

X_train.shape, y_train.shape
# ((90, 4), (90,))

X_test.shape, y_test.shape
# Out[9]: ((60, 4), (60,))


import pdb; pdb.set_trace()
iris.data[:5]

# # array([[ 5.1,  3.5,  1.4,  0.2],
#        [ 4.9,  3. ,  1.4,  0.2],
#        [ 4.7,  3.2,  1.3,  0.2],
#        [ 4.6,  3.1,  1.5,  0.2],
#        [ 5. ,  3.6,  1.4,  0.2]])

X_train[:5]

# array([[ 6. ,  3.4,  4.5,  1.6],
#        [ 4.8,  3.1,  1.6,  0.2],
#        [ 5.8,  2.7,  5.1,  1.9],
#        [ 5.6,  2.7,  4.2,  1.3],
#        [ 5.6,  2.9,  3.6,  1.3]])

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

clf.score(X_test, y_test)
# 0.96666666666666667
