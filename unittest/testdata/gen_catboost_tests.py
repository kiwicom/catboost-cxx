#!/usr/bin/env python3

import catboost as cb
import numpy as np
import numpy.matlib as mlib
import json
import os, sys

def xor_dataset():
    x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
    y = np.array([ 0, 1, 1, 0 ])
    return x, y

def or_dataset():
    x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
    y = np.array([ 0, 1, 1, 1 ])
    return x, y

def and_dataset():
    x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
    y = np.array([ 0, 0, 0, 1 ])
    return x, y

def regression_dataset():
    a = np.array( [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ] )
    x = np.random.rand(100, 8)
    y = mlib.dot(x, a) + np.random.rand(100) / 10.0
    return x, y

def gen_test(fnm, dataset, iterations = 10, learning_rate = 0.1, loss = "RMSE"):
    features, labels = dataset()
    model = cb.CatBoost({"learning_rate": learning_rate, "iterations": iterations, "loss_function": loss })
    model.fit(features, y = labels)
    model.save_model(fnm + "-model.json", format = "json")
    width = features.shape[1]
    x = np.random.rand(10, width)
    y = model.predict(x)
    with open(fnm + ".json", "wt") as f:
        json.dump({"x": x.tolist(), "y": y.tolist()}, f)

def main():
    gen_test("xor", xor_dataset, iterations = 50, learning_rate = 0.2, loss = "RMSE")
    gen_test("or", xor_dataset, iterations = 50, learning_rate = 0.2, loss = "RMSE")
    gen_test("and", xor_dataset, iterations = 50, learning_rate = 0.2, loss = "RMSE")
    gen_test("regression", regression_dataset, iterations = 100, learning_rate = 0.2, loss = "RMSE")


if __name__ == "__main__":
    main()
