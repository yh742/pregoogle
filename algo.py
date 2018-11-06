
# coding: utf-8

# In[200]:


#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import csv, time, sys
from textblob import TextBlob
import pandas
import sklearn
import random
import cPickle
import pylab as pl
import numpy as np
from sklearn.utils import shuffle
import textblob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve


# In[289]:


def clean(txt):
    not_list = ['...', 'you', 'your', 'his', 'her', 'will', "didn't", "i'll"]
    replace_list = [".", "'", '"', "!", "?", "#", "\r\n", "-", "@", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "$"]
    cleaned = []
    for token in str(txt).split(' '):
        flag = False
        for y in not_list:
            if y in token.lower():
                flag = True
                break
        if flag: continue
        for y in replace_list:
            token = token.replace(y, "")
        if len(token) > 3:
            token = token.strip().lower();
            try:
                #word = textblob.TextBlob(token).words[0]
                cleaned.append(token)
            except:
                continue
    return ' '.join(cleaned).decode("ascii", "ignore")


# In[290]:


def clean_pos(txt): 
    pos = TextBlob(txt).tags
    cleaned = [x[0] for x in pos if x[1] != 'IN' and x[1] != 'PRP' and x[1] != 'WP' and x[1] != 'FW']
    return ' '.join(cleaned)


# In[291]:


safe_stuff = []
no_safe_stuff = []


# In[418]:

f = open("trace.log", 'wb')
f.write("start\n")
print "sdf"

safedf = pandas.DataFrame()
no_safedf = pandas.DataFrame()
messages = pandas.read_csv('./safe_recipe.csv', sep=',', quoting=csv.QUOTE_NONE,
                           names=["date", "rating", "review"])
safedf['review'] = messages[messages.columns[2:4]] 
safedf['label'] = "safe"
safedf = shuffle(safedf).reset_index(drop=True)

messages = pandas.read_csv('./no_safe_recipe.csv', sep=',', quoting=csv.QUOTE_NONE,
                           names=["date", "rating", "review"])
no_safedf['review'] = messages[messages.columns[2:4]] 
no_safedf['label'] = "not_safe"

dataset = safedf.loc[0:10592,:]
dataset = dataset.append(no_safedf)
##print len(dataset)
dataset['review'] = dataset['review'].map(clean)
##print dataset

# In[397]:


dataset.replace('', np.nan, inplace=True)
dataset.dropna(subset=['review'], inplace=True)
dataset['review'] = dataset['review'].map(clean_pos)
##print dataset.head()

f.write("checkpoint1\n");
print "This is 1"
# In[398]:

#dataset.groupby('label').describe()


# In[399]:


dataset['length'] = dataset['review'].map(lambda text: len(text))
##print dataset.head()


# In[400]:


#dataset.length.plot(bins=20, kind='hist', figsize=(16, 8))


# In[401]:


#dataset.length.describe()


# In[402]:


#dataset.hist(column='length', by='label', bins=50, figsize=(16, 8))


# In[403]:


def split_into_lemmas(message):
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


# In[404]:


# not appropriate data, need training and test set
dataset = shuffle(dataset)


# In[405]:


msg_train, msg_test, label_train, label_test =     train_test_split(dataset['review'], dataset['label'], test_size=0.2)

##print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)


# In[406]:

f.write("checkpoint2\n")
print "This is 2"
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[407]:


# Cross-validation is a technique for evaluating ML models by training several 
# ML models on subsets of the available input data and
# evaluating them on the complementary subset of the data
scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=1,  # -1 = use all cores = faster
                         )
##print scores


# In[408]:


##print scores.mean(), scores.std()

print "after cv"
# In[409]:


params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas,),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)
nb_detector = grid.fit(msg_train, label_train)
f.write("checkpoint3\n")
print 'This is 3'

# In[410]:


#def ##print_top10(vectorizer, clf):
  #  """##prints features with the highest coefficient values, per class"""
  #  feature_names = vectorizer.get_feature_names()
    #for i, class_label in enumerate(class_labels):
    ##print np.argsort(clf.coef_[0])
 #   top10 = np.argsort(clf.coef_[0])[-10:]
 #   bot10 = np.argsort(clf.coef_[0])[0:10]
    ##print("%s" % (" ".join(feature_names[j] for j in top10)))
    ##print("%s" % (" ".join(feature_names[j] for j in bot10)))


###print nb_detector.grid_scores_
###print nb_detector.best_estimator_
##print_top10(best.named_steps['bow'], best.named_steps['classifier'])


# In[411]:


#predictions = nb_detector.predict(msg_test)
#prob = nb_detector.predict_proba(msg_test)


# In[430]:


##print predictions


def get_prediction(txt):
    blob = clean(txt)
    blob = clean_pos(blob)
    x = []
    x.append(blob)
    return nb_detector.predict(x)[0]

def get_prediction2(txt):
    blob = clean(txt)
    blob = clean_pos(blob)
    predi = nb_detector.predict_proba(txt)
    f.write(str(predi))
    predi = predi[:,1]
    f.write(str(predi))
    score = (np.sum(predi) / predi.size)
    f.write(str(score))
    rscore = round(score)
    f.write(str(rscore))
    if rscore == 1:
        return "safe"
    else:
        return "not_safe"

# In[432]:
f.write("checkpoint4\n")
print 'This is 4'

##print classification_report(label_test, predictions)
#print 'This is 5'





