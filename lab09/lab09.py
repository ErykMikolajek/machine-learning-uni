# %%
from sklearn.datasets import load_iris 
iris = load_iris(as_frame=True)

# %%
import pandas as pd

pd.concat([iris.data, iris.target], axis=1).plot.scatter(
    x='petal length (cm)', y='petal width (cm)', c='target',
    colormap='viridis', figsize=(10,4)
)

# %%
X = iris.data
X = X[["petal length (cm)", "petal width (cm)"]]
y = iris.target

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train

# %%
y_train0 = y_train.replace({0: 1, 1: 0, 2: 0})
y_train1 = y_train.replace({2: 0})
y_train2 = y_train.replace({1: 0, 2: 1})

y_test0 = y_test.replace({0: 1, 1: 0, 2: 0})
y_test1 = y_test.replace({2: 0})
y_test2 = y_test.replace({1: 0, 2: 1})

# %%
from sklearn.linear_model import Perceptron

per_clf0 = Perceptron()
per_clf0.fit(X_train, y_train0)

per_clf1 = Perceptron()
per_clf1.fit(X_train, y_train1)

per_clf2 = Perceptron()
per_clf2.fit(X_train, y_train2)

# %%
weights = [
    (*per_clf0.intercept_, per_clf0.coef_.tolist()[0][0], per_clf0.coef_.tolist()[0][1]),
    (*per_clf1.intercept_, per_clf1.coef_.tolist()[0][0], per_clf1.coef_.tolist()[0][1]),
    (*per_clf2.intercept_, per_clf2.coef_.tolist()[0][0], per_clf2.coef_.tolist()[0][1])
]
weights

# %%
from sklearn.metrics import accuracy_score

predicted_train_clf0 = per_clf0.predict(X_train)
predicted_test_clf0 = per_clf0.predict(X_test)

predicted_train_clf1 = per_clf1.predict(X_train)
predicted_test_clf1 = per_clf1.predict(X_test)

predicted_train_clf2 = per_clf2.predict(X_train)
predicted_test_clf2 = per_clf2.predict(X_test)

acc_list = [
    (accuracy_score(y_train0, predicted_train_clf0), accuracy_score(y_test0, predicted_test_clf0)),
    (accuracy_score(y_train1, predicted_train_clf1), accuracy_score(y_test1, predicted_test_clf1)),
    (accuracy_score(y_train2, predicted_train_clf2), accuracy_score(y_test2, predicted_test_clf2))]
acc_list

# %%
import pickle
with open("per_acc.pkl", "wb") as acc_f:
    pickle.dump(acc_list, acc_f)

# %%
with open("per_wght.pkl", "wb") as wght_f:
    pickle.dump(weights, wght_f)

# %%
import numpy as np
X = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]])
y = np.array([0, 1, 1, 0])

# %%
per_xor = Perceptron()
per_xor.fit(X, y)
print(per_xor.intercept_, per_xor.coef_)
print(per_xor.predict(X))

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
from sklearn.neural_network import MLPClassifier

activations_params = ['identity', 'logistic', 'tanh', 'relu']
solver_params = ['lbfgs', 'sgd', 'adam']
max_iter_params = range(1, 100)
learning_rate_params = ['constant', 'invscaling', 'adaptive']

win_models = []

for i_activation in activations_params:
    for i_solver in solver_params:
        for i_max_iter in max_iter_params:
            for i_learning_rate in learning_rate_params:
                clf_mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(2,), activation=i_activation, solver=i_solver, learning_rate=i_learning_rate, max_iter=i_max_iter).fit(X, y)
                if clf_mlp.score(X, y) >= 0.75:
                    win_models.append(clf_mlp)
win_models
print(win_models[0].intercepts_)
print(win_models[0].coefs_)
print(win_models[0].predict(X))
print(win_models[0].score(X, y))



# %%
with open("mlp_xor.pkl", "wb") as mlp_clf:
    pickle.dump(win_models[0], mlp_clf)

# %%
fixed_clf = MLPClassifier(hidden_layer_sizes=(2,)).fit(X, y)
fixed_clf.intercepts_ = [np.array([-1.5, -0.5]), np.array([-0.5])]
fixed_clf.coefs_ = [np.array([[0.0, 0.0], [0.0, 0.0]]), 
                    np.array([[-1.0], [0.0]])]
print(fixed_clf.intercepts_)
print(fixed_clf.coefs_)
print(fixed_clf.n_layers_)
print(fixed_clf.predict(X))
print(fixed_clf.score(X, y))

# %%
with open("mlp_xor_fixed.pkl", "wb") as xor_clf:
    pickle.dump(fixed_clf, xor_clf)

# %%
# fixed_clf.intercepts_ = [np.array([0.0, -1.0]), np.array([0.0])]
# fixed_clf.coefs_ = [np.array([[1.0, 1.0], [1.0, 1.0]]), 
#                     np.array([[1.0], [-2.0]])]


