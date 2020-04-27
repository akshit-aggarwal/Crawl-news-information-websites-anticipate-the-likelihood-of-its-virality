

import pandas
import numpy
import os
import matplotlib
import os
import argparse
from data_utils import read_clean_data, BinaryY
from models import (LinearRegression, LogisticRegression, 
					DecisionTree, SVM, NeuralNet,
					Bagging, RandomForest)
if 'scratch' in os.path.abspath('.'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
file_path = '------.csv'
NON_PREDICTIVE_COLS = ['url', 'timedelta', 'shares']
TARGET_COL = ['shares']
XY_IGNORE_COLS = ['url', 'timedelta']

def read_clean_data():
    full_data = clean_cols(pandas.read_csv(file_path))
    X = full_data[[x for x in list(full_data) if x not in NON_PREDICTIVE_COLS]]
    Y = full_data[TARGET_COL]
    return X, Y

def readXYcombined():
    full_data = clean_cols(pandas.read_csv(file_path))
    XY = full_data[[x for x in list(full_data) if x not in XY_IGNORE_COLS]]
    return XY

def clean_cols(data):
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(index=str, columns=clean_col_map)

def BinaryY(Y):
    Y['shares'] = Y['shares'].map(lambda x: 1 if x >= 1400 else 0)
    return Y

def TrainTestSplit(X, Y, R=0, test_size=0.2):
    return train_test_split(X, Y, test_size=test_size, random_state=R)

def PCAVisual(X, Y):
    Y = BinaryY(Y)
    
def main(grid):
	# Get Clean Data
	X, Y = read_clean_data()
	# Linear Regression
	try:
		LinearRegression(X, Y, grid)
	except Exception as e:
		print(e)
	# Binarize Y
	Y_binary = BinaryY(Y)
	# Logistic Regression
	try:
		LogisticRegression(X, Y_binary, grid)
	except Exception as e:
		print(e)
	# Decision Tree
	try:
		DecisionTree(X, Y_binary, grid)
	except Exception as e:
		print(e)
	# Support Vector Machine
	try:
		SVM(X, Y_binary, grid)
	except Exception as e:
		print(e)
	# Random Forest
	try:
		RandomForest(X, Y_binary, grid)
	except Exception as e:
		print(e)
	# Bagging Classifier
	try:
		Bagging(X, Y_binary, grid)
	except Exception as e:
		print(e)
	# Neural Network
	try:
		NeuralNet(X, Y_binary, grid)
	except Exception as e:
		print(e)
        
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid", help="Enable search.", action="store_true")
	args = parser.parse_args()
	main(args.grid)
    
