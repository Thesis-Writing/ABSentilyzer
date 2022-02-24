'''
  This module is intended for the classification class used in the performance
  evalutation of aspect term polarity classification.
'''

# Author            : Afrahly Afable
# Calling Sequence  : Classifier(train_text_list=None,train_aspect_dict_list=None,
#                                 test_text_list=None,test_aspect_dict_list=None,
#                                 test_aspect_list=None,polarity="pos"or"neg"or"neu",
#                                 mode="test"or"impelement")
# Date Written      : December 1, 2021
# Date Revised      : December 27, 2021
# Purpose           : Classify aspect term polarity
# Data Structures   : Input Variable/s:
#                       - train_text_list        : LIST
#                       - train_aspect_dict_list : LIST
#                       - test_text_list         : LIST
#                       - test_aspect_dict_list  : LIST
#                       - test_aspect_list       : LIST
#                       - train_label            : LIST
#                       - test_label             : LIST
#                       - polarity               : STRING
#                       - mode                   : STRING
#                     Output Variable/s:
#                       - atpc_eval_dict     : DICTIONARY
#                       - ensemble_pred_list : LIST
#                       - ensemble_prob_list : LIST
#                       

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.svm import SVC
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from ensemble_analyzer.apps.public.scripts.utils import *
from ensemble_analyzer.apps.public.scripts.metrics import *
from ensemble_analyzer.apps.public.scripts.display import *
from .ensemble import ensemble
from ensemble_analyzer.apps.public.models.mnb import MultinomialNB
from ensemble_analyzer.apps.public.models.ovr import OneVsRestClassifier


class Classifier:

  def __init__(self, train_text_list=None, train_aspect_dict_list=None,
      test_text_list=None, test_aspect_dict_list=None, test_aspect_list=None,
      polarity="pos"or"neg"or"neu", mode="test"or"impelement", 
      train_sent_label_list=None, test_sent_label_list=None):
    self.train_text_list = train_text_list
    self.train_aspect_dict_list = train_aspect_dict_list
    self.test_text_list = test_text_list
    self.test_aspect_dict_list = test_aspect_dict_list
    self.test_aspect_list = test_aspect_list
    self.polarity = polarity
    self.mode = mode
    self.train_sent_label_list = train_sent_label_list
    self.test_sent_label_list = test_sent_label_list

  def classify(self):
    '''
      This function acts as the main method of the Classifier class
    '''
    
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      display_header(self.polarity)

      y_test_list = []
      svm_pred_list = []
      svm_prob_list = []
      mnb_pred_list = []
      mnb_prob_list = []
      ensemble_pred_list = []
      ensemble_prob_list = []

      for i in tqdm(range(len(self.test_text_list))):
        test_text = []
        actual_aspect_dict_list = []
        target_aspect_list = self.test_aspect_list[i]

        test_text.append(self.test_text_list[i])
        actual_aspect_dict_list.append(self.test_aspect_dict_list[i])

        (X_train, y_train, 
        X_test, y_test) = self.get_data_model(self.train_text_list,
                                              self.train_aspect_dict_list,
                                              target_aspect_list,
                                              test_text,
                                              actual_aspect_dict_list,
                                              self.polarity)

        (y_pred_mnb, mnb_proba, 
            mnb_accuracy) = self.train_test_mnb(X_train, y_train, 
                                                X_test, y_test)
        (y_pred_svm, svm_proba, 
            svm_accuracy) = self.train_test_svm(X_train, y_train, 
                                                X_test, y_test)
        (ensemble_pred, ensemble_prob) = ensemble(y_pred_svm, y_pred_mnb,
                                                  svm_proba, mnb_proba)

        if self.mode == "test":
          if isinstance(y_test.tolist()[0], list):
            y_test_list.append(y_test.tolist()[0])
          else:
            y_test_list.append(y_test.tolist())

        if isinstance(y_pred_svm.tolist()[0], list):
          svm_pred_list.append(y_pred_svm.tolist()[0])
          svm_prob_list.append(svm_proba.tolist()[0])
          mnb_pred_list.append(y_pred_mnb.tolist()[0])
          mnb_prob_list.append(mnb_proba.tolist()[0])
          ensemble_pred_list.append(ensemble_pred.tolist()[0])
          ensemble_prob_list.append(ensemble_prob.tolist()[0])
        else:
          svm_pred_list.append(y_pred_svm.tolist())
          svm_prob_list.append(svm_proba.tolist())
          mnb_pred_list.append(y_pred_mnb.tolist())
          mnb_prob_list.append(mnb_proba.tolist())
          ensemble_pred_list.append(ensemble_pred.tolist())
          ensemble_prob_list.append(ensemble_prob.tolist())

      if self.mode == "test":
        (svm_accuracy, svm_precision, 
            svm_recall, svm_f1) = get_al_metrics(svm_pred_list, 
                                                y_test_list, "micro",
                                                mode="model",
                                                classifier="SVM")
        (mnb_accuracy, mnb_precision, 
            mnb_recall, mnb_f1) = get_al_metrics(mnb_pred_list, 
                                                y_test_list, "micro",
                                                mode="model",
                                                classifier="MNB")
        (ensemble_accuracy, 
            ensemble_precision, 
            ensemble_recall, 
            ensemble_f1) = get_al_metrics(ensemble_pred_list, 
                                            y_test_list, "micro",
                                            mode="model",
                                            classifier="ENSEMBLE")

        display_footer(self.polarity)
        display_model_performance(svm_accuracy,svm_precision,svm_recall,svm_f1,
                                  mnb_accuracy,mnb_precision,mnb_recall,mnb_f1,
                                  ensemble_accuracy,ensemble_precision,
                                  ensemble_recall, ensemble_f1)

        atpc_eval_dict = {'SVM': [svm_accuracy,
                                  svm_precision,
                                  svm_recall,
                                  svm_f1],
                          'MNB': [mnb_accuracy,
                                  mnb_precision,
                                  mnb_recall,
                                  mnb_f1],
                          'Ensemble SVM-MNB': [ensemble_accuracy,
                                              ensemble_precision,
                                              ensemble_recall,
                                              ensemble_f1]}

        return (ensemble_prob_list, atpc_eval_dict)
      elif self.mode == "implement":
        return (ensemble_prob_list)

  def get_data_model(self, train_text_list=None, train_aspect_dict_list=None, 
                    target_aspect_list=None, test_text=None, 
                    test_aspect_dict_list=None, polarity=None):
    '''
      This function returns the data model that will be used in the 
      classification
    '''

    if polarity == None:

      df_train = get_sentence_data_frame(self.train_text_list, 
                                        self.train_sent_label_list)
      df_test = get_sentence_data_frame(self.test_text_list, 
                                        self.test_sent_label_list)

      (X_train, y_train, 
        X_test, y_test) = self.preprare_data_model(df_train, df_test)

    else:
      
      df_train = get_data_frame(train_text_list, train_aspect_dict_list, 
                                target_aspect_list)
      df_test = get_data_frame(test_text, test_aspect_dict_list, 
                              target_aspect_list)
      if polarity == "pos":
        df_train = get_positive_data_frame(df_train, target_aspect_list)
        df_test = get_positive_data_frame(df_test, target_aspect_list)
      elif polarity == "neg":
        df_train = get_negative_data_frame(df_train, target_aspect_list)
        df_test = get_negative_data_frame(df_test, target_aspect_list)
      elif polarity == "neu":
        df_train = get_neutral_data_frame(df_train, target_aspect_list)
        df_test = get_neutral_data_frame(df_test, target_aspect_list)

      (X_train, y_train, 
          X_test, y_test) = self.preprare_data_model(df_train,df_test)

      X_train = self.get_extra_features(X_train, y_train, target_aspect_list,
                                        train_aspect_dict_list, polarity)

      X_test = self.get_extra_features(X_test, y_test, target_aspect_list,
                                      test_aspect_dict_list, polarity)

    return (X_train, y_train, X_test, y_test)

  def preprare_data_model(self, data_train: pd.DataFrame,
                          data_test: pd.DataFrame):
    '''
      This function returns the sparse matrix format of the train 
      and test data

      Data Structures
      ---------------
      Input:
        data_train    : DATA FRAME
        data_test     : DATA FRAME
        target_aspect : LIST
      Returns:
        X_train       : SPARSE MATRIX
        y_train       : SPARSE MATRIX
        X_test        : SPARSE MATRIX
        y_test        : SPARSE MATRIX
    '''
    
    X_train = data_train.tweets
    y_train = data_train.drop('tweets',1)
    y_train = np.array(y_train, dtype=np.int64)

    X_test = data_test.tweets
    y_test = data_test.drop('tweets',1)
    y_test = np.array(y_test, dtype=np.int64)

    tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                            lowercase=False, max_features=5000, 
                            ngram_range=(1,4))
    tfidf.fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)

    return (X_train, y_train, X_test, y_test)

  def get_extra_features(self, X, y, target_aspect_list, 
                        aspect_dict_list, polarity):

    '''
      This function returns the extra features used in the
      sparse matrix of both training and testing vectors
    '''

    dictionary_vect=DictVectorizer()

    try:
      dict_aspect = get_dict_aspect(y, target_aspect_list)
      X_aspect_dtm = dictionary_vect.fit_transform(dict_aspect)

      dict_aspect = get_dict_aspect_polarity(y, target_aspect_list, 
                                            aspect_dict_list, 
                                            polarity=polarity)
      X_class_aspect_dtm = dictionary_vect.fit_transform(dict_aspect)
      
      X = hstack((X, X_aspect_dtm))
      X = hstack((X, X_class_aspect_dtm))
      
      return X
    except:
      print("Invalid polarity")
      
      return None

  def train_test_mnb(self, X_train, y_train, X_test, y_test):
    '''
      This function is used to fit the MNB model with the training data using
      and then predict the test data using the fitted model

      Data Structures
      ---------------
      Input:
        X_train       : SPARSE MATRIX
        y_train       : SPARSE MATRIX
        X_test        : SPARSE MATRIX
        y_test        : SPARSE MATRIX
      Returns:
        y_pred_mnb    : NUMPY ARRAY
        mnb_proba     : NUMPY ARRAY
        mnb_accuracy  : FLOAT
    '''
    model = OneVsRestClassifier(MultinomialNB(alpha=1))
    mnb_model = model.fit(X_train, y_train)
    y_pred_mnb = mnb_model.predict(X_test)
    mnb_proba = mnb_model.predict_proba(X_test)
    mnb_accuracy = mnb_model.score(X_test, y_test)

    return (y_pred_mnb,mnb_proba,mnb_accuracy)

  def train_test_svm(self, X_train, y_train, X_test, y_test):
    '''
      This function is used to fit the SVM model with the training data using
      and then predict the test data using the fitted model

      Data Structures
      ---------------
      Input:
        X_train       : SPARSE MATRIX
        y_train       : SPARSE MATRIX
        X_test        : SPARSE MATRIX
        y_test        : SPARSE MATRIX
      Returns:
        y_pred_svm    : NUMPY ARRAY
        svm_proba     : NUMPY ARRAY
        svm_accuracy  : FLOAT
    '''

    model = OneVsRestClassifier(SVC(C=10, kernel="linear", probability=True))
    svm_model = model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_proba = svm_model.predict_proba(X_test)
    svm_accuracy = svm_model.score(X_test, y_test)

    return (y_pred_svm, svm_proba, svm_accuracy)

  def classify_sentence(self):
    '''
      This function performs the sentence level classification using the 
      machine learning models only
    '''
    
    (X_train, y_train, 
        X_test, y_test) = self.get_data_model()

    (y_pred_mnb, mnb_proba, 
        mnb_accuracy) = self.train_test_mnb(X_train, y_train, X_test, y_test)
    (y_pred_svm, svm_proba, 
        svm_accuracy) = self.train_test_svm(X_train, y_train, X_test, y_test)
    (ensemble_pred, 
        ensemble_prob) = ensemble(y_pred_svm, y_pred_mnb, svm_proba, mnb_proba)

    y_test = encode_label(y_test)
    y_pred_svm = encode_label(y_pred_svm)
    y_pred_mnb = encode_label(y_pred_mnb)
    ensemble_pred = encode_label(ensemble_pred)

    (svm_accuracy, svm_precision,
        svm_recall, svm_f1, s_tp, 
        s_fp, s_tn, s_fn) = get_sl_metrics(y_test, y_pred_svm, 
                                          classifier="SVM")
    (mnb_accuracy, mnb_precision,
        mnb_recall, mnb_f1, m_tp, 
        m_fp, m_tn, m_fn) = get_sl_metrics(y_test, y_pred_mnb, 
                                          classifier="MNB")
    (ensemble_accuracy, 
        ensemble_precision, 
        ensemble_recall, 
        ensemble_f1,
        e_tp, e_fp, e_tn, e_fn) = get_sl_metrics(y_test, ensemble_pred, 
                                                classifier="ENSEMBLE")

    print("==============================================================")
    print("           SENTENCE LEVEL WITHOUT ASPECT EVALUATION           ")
    print("==============================================================")
    display_model_performance(svm_accuracy, svm_precision, svm_recall, svm_f1, 
                              mnb_accuracy, mnb_precision, mnb_recall, mnb_f1,
                              ensemble_accuracy, ensemble_precision, 
                              ensemble_recall, ensemble_f1)
    print("\nSVM-TP: {}, SVM-FP: {}, SVM-TN: {}, SVM-FN: {}".format(s_tp, s_fp, 
                                                                  s_tn, s_fn))
    print("MNB-TP: {}, MNB-FP: {}, MNB-TN: {}, MNB-FN: {}".format(m_tp, m_fp, 
                                                                  m_tn, m_fn))
    print("ENS-TP: {}, ENS-FP: {}, ENS-TN: {}, ENS-FN: {}".format(e_tp, e_fp, 
                                                                  e_tn, e_fn))