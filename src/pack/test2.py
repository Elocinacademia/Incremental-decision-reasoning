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

print('test')
