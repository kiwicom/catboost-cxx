#!/usr/bin/env python3

import catboost as cb
import catboost.datasets as ds
import tensorflow as tf
import sklearn.datasets
import sklearn.model_selection
import numpy as np

import sys
import shutil
from collections import namedtuple


Dataset = namedtuple("Dataset", [ "train_x", "train_y", "test_x", "test_y" ])


class ModelBase:
    ITERATIONS = 1000
    LEARNING_RATE = 0.05
    LOSS = "RMSE"

    def dataset(self):
        raise NotImplemented

    def create(self):
        return cb.CatBoostRegressor(self.ITERATIONS, learning_rate = self.LEARNING_RATE, loss_function = self.LOSS)

    def train(self, dataset):
        model = self.create()
        l_ds = cb.Pool(
                data = dataset.train_x,
                label = dataset.train_y
        )
        t_ds = cb.Pool(
                data = dataset.test_x,
                label = dataset.test_y
        )
        model.fit(l_ds, eval_set = t_ds)
        return model

    def name(self):
        cls = self.__class__.__name__
        assert cls.endswith("Model")
        return cls[:-5].lower()


def split_dataset(ds):
    count = ds.shape[1]
    data = ds[[i for i in range(1, count)]]
    label = ds[[0]]
    return data, label


MODELS = []
def model(cls):
    MODELS.append(cls)
    return cls

@model
class MSRankModel(ModelBase):
    ITERATIONS = 5000
    # ITERATIONS = 100
    LEARNING_RATE = 0.01

    def dataset(self):
        d = ds.msrank()
        l_x, l_y = split_dataset(d[0])
        t_x, t_y = split_dataset(d[1])

        return Dataset(l_x, l_y, t_x, t_y)


@model
class CreditGermanyModel(ModelBase):
    ITERATIONS = 1000
    LEARNING_RATE = 0.01
    LOSS = "Quantile"

    def dataset(self):
        ds = sklearn.datasets.fetch_openml("credit-g")
        x = None
        y = None

        x = np.array(ds["data"])
        y = np.array([[1.0] if l == 'good' else [0.0] for l in ds["target"]])

        # Now we need to split dataset into learn/test
        l_x, t_x, l_y, t_y = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
        return Dataset(l_x, l_y, t_x, t_y)

@model
class CodRNAModel(ModelBase):
    ITERATIONS = 1000
    LEARNING_RATE = 0.01
    LOSS = "Quantile"

    def dataset(self):
        ds = sklearn.datasets.fetch_openml("codrnaNorm")
        x = None
        y = None

        x = ds["data"].toarray()
        y = np.array([[float(a)] for a in ds["target"]])

        # Now we need to split dataset into learn/test
        l_x, t_x, l_y, t_y = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
        return Dataset(l_x, l_y, t_x, t_y)

def save_tsv(x, y, filename):
    x = np.array(x) # .to_numpy()
    y = np.array(y) # .to_numpy()
    with open(filename, "wt", encoding = "utf-8") as f:
        cnt = x.shape[0]
        for i in range(cnt):
            f.write("%f\t%s\n" % (y[i][0], "\t".join([str(v) for v in x[i]])))


def train_one(Class, copy = False):
    model = Class()
    dataset = model.dataset()
    m = model.train(dataset)
    fnm = model.name()
    m.save_model(fnm + ".cbm", format = "cbm")
    m.save_model(fnm + ".json", format = "json")
    m.save_model(fnm + "-orig.cpp", format = "cpp")
    with open(fnm + "-orig.cpp", "rt", encoding = "utf-8") as f:
        cpp = f.read()
    with open(fnm + ".cpp", "wt", encoding = "utf-8") as f:
        f.write(cpp.replace("CatboostModel", model.__class__.__name__))
        f.write(f"""struct Static{model.__class__.__name__} {{
    double predict(const std::vector<float>& x) const {{
        return Apply{model.__class__.__name__}(x);
    }}

    void predict(const std::vector<std::vector<float>>& x, std::vector<double>& y) const {
        y.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = predict(x[i]);
    }
}};
""")

    save_tsv(dataset.test_x, dataset.test_y, fnm + "_test.tsv")

    if copy:
        for nm in [ fnm + ".cbm", fnm + ".json", fnm + ".cpp", fnm + "_test.tsv" ]:
            shutil.copy(nm, "../" + nm)

def main():
    if len(sys.argv) <= 1:
        print("Training all models...")
        for Class in MODELS:
            train_one(Class, True)
    elif sys.argv[1] == 'list':
        for m in MODELS:
            print(f"{m.__name__}: {m().name()}")
    else:
        for nm in sys.argv[1:]:
            Class = None
            for cls in MODELS:
                if cls().name() == nm:
                    Class = cls
                    break
            if Class is None:
                print(f"Error: Unknown model: {nm}")
                sys.exit(1)
            train_one(Class, False)

if __name__ == '__main__':
    main()

