
# coding: utf-8

# https://github.com/wallinm1/kaggle-facebook-bot/blob/master/facebook_notebook.ipynb

# # Nanodegree Engenheiro de Machine Learning
# 
# ## Capstone Project
# 
# Este projeto consiste em classificar em uma base histórica de um site de leilões os lances realizados de forma automatizada (feita por robôs) e aqueles feitos pelos usuários tradicionais (humanos).

# In[1]:


import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

get_ipython().magic(u'matplotlib inline')


# ## Importando os dados
# 
# Abaixo vamos importar as bases de treino e lances, em seguida faremos o merge entre elas.

# In[2]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[3]:


bids = pd.read_csv('../input/bids.csv')
bids.head()


# In[4]:


train_bids = pd.merge(bids, train, how='left', on='bidder_id')
train_bids.head()


# Na célula abaixo são removidos os registros onde não constam a variável resposta *outcome* e a variável *country*. Esta abordagem foi adotada em preferência a preencher estes valores faltantes para não criar distorções no dataset.

# In[5]:


train_bids.dropna(subset=['outcome'], inplace=True)
train_bids.dropna(subset=['country'], inplace=True)
train_bids.head()


# In[6]:


del train
del bids


# Abaixo confirmamos a proporção do desbalanceamento do dataset. Apenas 13,4% dos lances foram feitos por robôs.

# In[7]:


to_plot = train_bids.groupby('outcome')['bid_id'].count()
to_plot = to_plot.groupby(level=0).apply(lambda x: 100 * x / train_bids.shape[0]).reset_index()
print(to_plot)
ax = sns.barplot(x="outcome", y="bid_id", data=to_plot);
ax.set(ylabel="Percent");


# Abaixo vemos as proporções de lances feitos por usuários e robôs por cada categoria de produto leiloado indicando que esta pode ser uma variável importante para o modelo.
# 
# Outras visualizações foram geradas mas as mesmas não agregaram mais entendimento dos dados e por isso foram removidas.

# In[8]:


to_plot = train_bids.groupby(['outcome', 'merchandise'])['bid_id'].count()
fig, ax = plt.subplots(figsize=(15, 7))
sns.barplot(x='merchandise', y='bid_id', hue='outcome', data=to_plot.reset_index(), ax=ax);


# Abaixo vamos preparar os dados categóricos da base para alimentar um modelo de árvore de decisão que nos ajudará a entender a importancia de cada variável na categorização dos lances. Usaremos a classe *LabelEncoder* do pacote *scikit-learn* para isso.

# In[9]:


X = train_bids.copy()
X.drop(['bid_id', 'outcome', 'time'], axis=1, inplace=True)

d = defaultdict(LabelEncoder)
X = X.apply(lambda x: d[x.name].fit_transform(x))
X.head()


# Abaixo faremos o ajuste da variável alvo e separaremos o dataset em conjuntos de treino e teste. Usaremos o conjunto de teste para avaliar o modelo gerado.

# In[10]:


y = np.ravel(train_bids.outcome)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Abaixo vamos confirmar se os conjuntos de treino e teste preservaram as proporções do dataset original

# In[12]:


from __future__ import division
from collections import Counter

print("treino")
x = Counter(y_train)
print(x[0.0]/len(y_train))
print(x[1.0]/len(y_train))

print("teste")
x = Counter(y_test)
print(x[0.0]/len(y_test))
print(x[1.0]/len(y_test))


# Abaixo será feito o treino do modelo de árvore de decisão. Este modelo nos dirá a importância de cada variável do dataset e será usado para uma primeira avaliação da nossa capacidade de classificação.

# In[13]:


forest = ExtraTreesClassifier(
    n_estimators=250,
    criterion="entropy",
    min_samples_split=30,
    max_depth=10,
    random_state=42,
    n_jobs=4)
forest.fit(X_train, y_train)


# Abaixo são mostradas as variáveis e suas importâncias em uma visualização em barras. As variáveis *address*, *bidder_id*, *payment_account* e *merchandise* foram identificadas como as mais importantes para o modelo.

# In[14]:


importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
feature_names = [X.columns[i] for i in indices]

plt.figure(figsize=(15,6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[15]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Abaixo o modelo de árvore de decisão é testado usando matriz de confusão, precision score e recall score.

# In[16]:


class_names = ['human', 'robot']
y_pred = forest.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=4)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')


# In[17]:


y_scores = forest.predict_proba(X_test)[:,1]

print("Precision Score: %s" % precision_score(y_test, y_pred))
print("Recall Score: %s" % recall_score(y_test, y_pred))
print("ROC AUC Score: %s" % roc_auc_score(y_test, y_scores))


# Abaixo faremos a plotagem da curva ROC que é usada no desafio como método de avaliação

# In[18]:


from sklearn.metrics import roc_curve, auc

# ROC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Aparentemente o modelo treinado para nos informar as variáveis mais importantes para o modelo conseguiu um bom desempenho e poderia ser usado para classificar os lances, mas ao submeter os resultados obtidos à competição na plataforma Kaggle o score obtido foi de 50,7%
# 
# ![kaggle](kaggle.png)

# Como sugerido na proposta de projeto, vamos comparar o desempenho deste modelo treinado com um classificador "Dummy".

# In[19]:


dummy = DummyClassifier(random_state=42)
dummy.fit(X_train, y_train)
dummy_predict = dummy.predict(X_test)

class_names = ['human', 'robot']
cnf_matrix = confusion_matrix(y_test, dummy_predict)
np.set_printoptions(precision=4)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')


# In[20]:


print("Precision Score: %s" % precision_score(y_test, dummy_predict))
print("Recall Score: %s" % recall_score(y_test, dummy_predict))
print("ROC AUC Score: %s" % roc_auc_score(y_test, dummy_predict))


# In[21]:


y_scores = dummy.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Os testes feitos até aqui reaproveitaram um modelo que foi treinado com o objetivo de avaliar as features disponíveis. Vamos agora treinar outros algoritmos para vermos os resultados.

# In[22]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[23]:


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3, n_jobs=4),
    SVC(kernel="linear", C=0.025, class_weight=[{0.0:0.86, 1.0:0.13}]),
    SVC(gamma=2, C=1, class_weight=[{0.0:0.86, 1.0:0.13}]),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=4),
    DecisionTreeClassifier(max_depth=5, class_weight=[{0.0:0.86, 1.0:0.13}]),
    RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=4),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]


# In[24]:


results = []
for name, clf in zip(names, classifiers):
    print(name)
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)[:,1]
    except:
        print "Could not train %s model" % name
        continue

    results.append({
        'name': name,
        'precision': precision_score(y_test, y_pred),
        'recal': recall_score(y_test, y_pred),
        'ROC score': roc_auc_score(y_test, y_scores)
    }) 


# ## Checkpoint
# 
# Salvandos os dados para poder prosseguir daqui.

# In[30]:


import pickle

with open('../input/train_test.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)

with open('../input/results.pkl', 'wb') as f:
    pickle.dump(results, f)


# Carregando os dados

# In[33]:


import pickle

with open('../input/train_test.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

with open('../input/results.pkl', 'rb') as f:
    results = pickle.load(f)


# In[44]:


results = pd.DataFrame(results)
results = results.set_index('name')


# In[49]:


results['mean'] = results.mean(axis=1)
results

