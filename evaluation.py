import pandas as pd
import spacy
import re
import warnings
from spacy.lang.char_classes import LIST_PUNCT
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
nlp = spacy.load("en_core_web_sm")

svm_linear = svm.SVC(probability=True, random_state = 123)
nb = GaussianNB()
rf = RandomForestClassifier(n_jobs = -1, random_state = 123)
xgb = XGBClassifier(eval_metric = 'logloss', random_state = 123)
soft_vote = VotingClassifier(estimators = [('svm', svm_linear), ('nb', nb), ('rf', rf), ('xgb',xgb)], voting = 'soft', n_jobs=-1) 
hard_vote = VotingClassifier(estimators = [('svm', svm_linear), ('nb', nb), ('rf', rf), ('xgb',xgb)], voting = 'hard', n_jobs=-1) 
clf_list = [svm_linear, nb, rf, xgb, soft_vote, hard_vote]
clf_names = ["Linear SVM", "Naive Bayes", "Random Forest", "XGBoost", "Soft Voting", "Hard Voting"]


def avg_score_list(X_train, y_train):
    cv = 5
    cvs_acc = [cross_val_score(clf, X_train, y_train, cv = cv, scoring = 'accuracy', n_jobs=-1).mean() for clf in clf_list]
    cvs_prec = [cross_val_score(clf, X_train, y_train, cv = cv, scoring = 'precision', n_jobs=-1).mean() for clf in clf_list]
    cvs_rec = [cross_val_score(clf, X_train, y_train, cv = cv, scoring = 'recall', n_jobs=-1).mean() for clf in clf_list]
    cvs_f1 = [cross_val_score(clf, X_train, y_train, cv = cv, scoring = 'f1', n_jobs=-1).mean() for clf in clf_list]
    return cvs_acc, cvs_prec, cvs_rec, cvs_f1


def score_df(X_train, y_train) -> pd.DataFrame:
    allscore_df = pd.DataFrame()
    allscore_df["Classifier"] = clf_names
    allscore_df["Average_accuracy_score"] = avg_score_list(X_train, y_train)[0]
    allscore_df["Average_precision_score"] = avg_score_list(X_train, y_train)[1]
    allscore_df["Average_recall_score"] = avg_score_list(X_train, y_train)[2]
    allscore_df["Average_f1-score"] = avg_score_list(X_train, y_train)[3]
    return allscore_df


