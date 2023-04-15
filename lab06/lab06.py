#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[3]:


import pandas as pd
cancer_X = pd.DataFrame(data=data_breast_cancer['data'], columns=data_breast_cancer['feature_names'])
cancer_X = cancer_X[['mean texture', 'mean symmetry']]
cancer_y = pd.DataFrame(data=data_breast_cancer['target'], columns=['target'])


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=0.2, random_state=42)


# In[5]:


from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dt_clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
lr_clf = LogisticRegression(random_state=42).fit(X_train, y_train)
knn_clf = KNeighborsClassifier().fit(X_train, y_train)

hard_voting_clf = VotingClassifier(
    estimators=[('dt', DecisionTreeClassifier(random_state=42)),
                ('lr', LogisticRegression(random_state=42)),
                ('knn', KNeighborsClassifier())],
    voting='hard')

soft_voting_clf = VotingClassifier(
    estimators=[('dt', DecisionTreeClassifier(random_state=42)),
                ('lr', LogisticRegression(random_state=42)),
                ('knn', KNeighborsClassifier())],
    voting='soft')


# In[6]:


hard_voting_clf.fit(X_train, y_train)
soft_voting_clf.fit(X_train, y_train)


# In[7]:


dt_predicted_train = dt_clf.predict(X_train)
dt_predicted_test = dt_clf.predict(X_test)

lr_predicted_train = lr_clf.predict(X_train)
lr_predicted_test = lr_clf.predict(X_test)

knn_predicted_train = knn_clf.predict(X_train)
knn_predicted_test = knn_clf.predict(X_test)

hard_voting_predicted_train = hard_voting_clf.predict(X_train)
hard_voting_predicted_test = hard_voting_clf.predict(X_test)

soft_voting_predicted_train = soft_voting_clf.predict(X_train)
soft_voting_predicted_test = soft_voting_clf.predict(X_test)


# In[8]:


from sklearn.metrics import accuracy_score
acc_list = []

acc_list.append((accuracy_score(y_train, dt_predicted_train), accuracy_score(y_test, dt_predicted_test)))
acc_list.append((accuracy_score(y_train, lr_predicted_train), accuracy_score(y_test, lr_predicted_test)))
acc_list.append((accuracy_score(y_train, knn_predicted_train), accuracy_score(y_test, knn_predicted_test)))
acc_list.append((accuracy_score(y_train, hard_voting_predicted_train), accuracy_score(y_test, hard_voting_predicted_test)))
acc_list.append((accuracy_score(y_train, soft_voting_predicted_train), accuracy_score(y_test, soft_voting_predicted_test)))
acc_list


# In[9]:


import pickle
with open('acc_vote.pkl', 'wb') as acc_pickle:
    pickle.dump(acc_list, acc_pickle)


# In[10]:


clf_list = []

clf_list.append(dt_clf)
clf_list.append(lr_clf)
clf_list.append(knn_clf)
clf_list.append(hard_voting_clf)
clf_list.append(soft_voting_clf)


# In[11]:


with open('vote.pkl', 'wb') as vote_pickle:
    pickle.dump(clf_list, vote_pickle)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), 
                            n_estimators=30, max_samples=1.0, bootstrap=True, random_state=42).fit(X_train, y_train)
bag_50_clf = BaggingClassifier(DecisionTreeClassifier(), 
                            n_estimators=30, max_samples=0.5, bootstrap=True, random_state=42).fit(X_train, y_train)
past_clf = BaggingClassifier(DecisionTreeClassifier(), 
                            n_estimators=30, max_samples=1.0, bootstrap=False, random_state=42).fit(X_train, y_train)
past_50_clf = BaggingClassifier(DecisionTreeClassifier(), 
                            n_estimators=30, max_samples=0.5, bootstrap=False, random_state=42).fit(X_train, y_train)
rnd_clf = RandomForestClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)
ada_clf = AdaBoostClassifier(n_estimators=30).fit(X_train, y_train)
grad_clf = GradientBoostingClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)


# In[13]:


bag_predicted_train = bag_clf.predict(X_train)
bag_predicted_test = bag_clf.predict(X_test)

bag_50_predicted_train = bag_50_clf.predict(X_train)
bag_50_predicted_test = bag_50_clf.predict(X_test)

past_predicted_train = past_clf.predict(X_train)
past_predicted_test = past_clf.predict(X_test)

past_50_predicted_train = past_50_clf.predict(X_train)
past_50_predicted_test = past_50_clf.predict(X_test)

rnd_predicted_train = rnd_clf.predict(X_train)
rnd_predicted_test = rnd_clf.predict(X_test)

ada_predicted_train = ada_clf.predict(X_train)
ada_predicted_test = ada_clf.predict(X_test)

grad_predicted_train = grad_clf.predict(X_train)
grad_predicted_test = grad_clf.predict(X_test)


# In[14]:


acc_list2 = []

acc_list2.append((accuracy_score(y_train, bag_predicted_train), accuracy_score(y_test, bag_predicted_test)))
acc_list2.append((accuracy_score(y_train, bag_50_predicted_train), accuracy_score(y_test, bag_50_predicted_test)))
acc_list2.append((accuracy_score(y_train, past_predicted_train), accuracy_score(y_test, past_predicted_test)))
acc_list2.append((accuracy_score(y_train, past_50_predicted_train), accuracy_score(y_test, past_50_predicted_test)))
acc_list2.append((accuracy_score(y_train, rnd_predicted_train), accuracy_score(y_test, rnd_predicted_test)))
acc_list2.append((accuracy_score(y_train, ada_predicted_train), accuracy_score(y_test, ada_predicted_test)))
acc_list2.append((accuracy_score(y_train, grad_predicted_train), accuracy_score(y_test, grad_predicted_test)))
acc_list2


# In[15]:


with open('acc_bag.pkl', 'wb') as acc_bag_pickle:
    pickle.dump(acc_list2, acc_bag_pickle)


# In[16]:


clf_list2 = []
clf_list.append(bag_clf)
clf_list.append(bag_50_clf)
clf_list.append(past_clf)
clf_list.append(past_50_clf)
clf_list.append(rnd_clf)
clf_list.append(ada_clf)
clf_list.append(grad_clf)


# In[17]:


with open('bag.pkl', 'wb') as bag_pickle:
    pickle.dump(clf_list2, bag_pickle)


# In[26]:


cancer_X_all_features = pd.DataFrame(data=data_breast_cancer['data'], columns=data_breast_cancer['feature_names'])
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(cancer_X_all_features, cancer_y, test_size=0.2, random_state=42)


# In[27]:


sampling = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                             bootstrap=True, bootstrap_features=False, max_features=2, 
                             max_samples=0.5, random_state=42).fit(X_train_all, y_train_all)


# In[28]:


sampling_predicted_train = sampling.predict(X_train_all)
sampling_predicted_test = sampling.predict(X_test_all)


# In[29]:


acc_list3 = []
acc_list3.append(accuracy_score(y_train_all, sampling_predicted_train))
acc_list3.append(accuracy_score(y_test_all, sampling_predicted_test))
acc_list3


# In[30]:


with open('acc_fea.pkl', 'wb') as acc_fea_pickle:
    pickle.dump(acc_list3, acc_fea_pickle)


# In[31]:


fea_list = [sampling]
with open('fea.pkl', 'wb') as fea_pickle:
    pickle.dump(fea_list, fea_pickle)


# In[118]:


estimators_df = pd.DataFrame()
i = 0
for estimator in sampling.estimators_:
    features = sampling.estimators_features_[i]
    estimator_predicted_train = estimator.predict(X_train_all.iloc[:, features])
    estimator_predicted_test = estimator.predict(X_test_all.iloc[:, features])
    new_row = pd.DataFrame([accuracy_score(y_train_all, estimator_predicted_train), 
                            accuracy_score(y_test_all, estimator_predicted_test), 
                            [name for name in cancer_X_all_features.columns[features]]]).T
    estimators_df = estimators_df.append(new_row)
    i += 1
estimators_df.columns=['Train_acc', 'Test_acc', 'Feature_names']
estimators_df


# In[121]:


estimators_df = estimators_df.sort_values(['Test_acc', 'Train_acc'], ascending = [False, False])
estimators_df


# In[122]:


with open('acc_fea_rank.pkl', 'wb') as fea_rank:
    pickle.dump(estimators_df, fea_rank)


# In[ ]:




