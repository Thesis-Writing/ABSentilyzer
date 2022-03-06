# Program Title         : absentilyzer
# Author                : Afrahly Afable
# General System Design : This class is intended for the processes used in the 
#                         aspect-based sentiment analysis .An instance of the 
#                         class is created in views.py with the user input as 
#                         parameter. The get_tables() function returns the 
#                         dictionaries used to display the contents
#                         in the frontend tables
# Date Written          : December 1, 2021
# Date Revised          : February 20, 2021
# Purpose               : To perform aspect-based sentiment 
#                          analysis on the user input
# Data Structures       : List, Dictionary, Object, Integer

import warnings
import pickle
import time
import os

from ensemble_analyzer.apps.public.scripts.utils import *
from ensemble_analyzer.apps.public.scripts.display import *
from ensemble_analyzer.apps.public.scripts.validation import check_tweet
from ensemble_analyzer.apps.public.scripts.polarity_calculation import *
from ensemble_analyzer.apps.public.scripts.preprocessing import preprocess
from ensemble_analyzer.apps.public.scripts.aspect_polarity_classifier import Classifier
from ensemble_analyzer.apps.public.scripts.aspect_term_extraction import AspectTermExtraction

cwd = os.getcwd()

class ABSentilyzer():
  def __init__(self, user_input):
      self.original_test_text_list = user_input

  def process_input(self):
    # Process user input
    with warnings.catch_warnings():
      self.start = time.time()
      warnings.simplefilter("ignore")
      
      self.get_train_data()
      self.preprocess_input()
      self.extract_aspect()
      self.classify_aspect_pol()
      self.aggregate_aspect_pol()
      self.get_tweet_pol()

  def get_tables(self):
    # Get all tables for display and print runtime in terminal
    self.get_main_table_dict()
    self.get_tweet_pol_table_dict()
    self.get_runtime()

    return (self.main_table_dict_list, self.tweet_pol_table_dict)

  def get_runtime(self):
    # Get AB Sentilyzer runtime
    end = time.time()
    time_linear_process = end-self.start
    print("\nStarted: %fs; \nEnded: %fs; \nSentiLyzer Process Time:%fs\n"
        % (self.start, end, time_linear_process))

  def get_train_data(self):
    # Get train data
    train_path = 'ensemble_analyzer/apps/public/data/train.pkl'
    (self.original_train_text_list,self.train_aspect_dict_list,
      self.train_label_list) = get_list_from_data_frame(train_path)

    preprocessed_train_list_path = 'ensemble_analyzer/apps/public/data/'\
                                    'preprocessed_train_list.pkl'
    self.preprocessed_train_text_list = pickle.loads(
      open(os.path.join(cwd, preprocessed_train_list_path), 'rb').read())

    return (self.original_train_text_list, self.preprocessed_train_text_list, 
            self.train_aspect_dict_list, self.train_label_list)

  def preprocess_input(self):
    self.preprocessed_test_text_list = preprocess(self.original_test_text_list)

  def extract_aspect(self):
    # Extract aspect terms from input tweets
    ate = AspectTermExtraction(self.preprocessed_test_text_list, 
                              mode="implement")
    (self.extracted_aspect_list, 
        self.opinion_aspect_dict_list, 
        self.dependency_list) = ate.get_extracted_aspects()
    self.remove_no_aspect_tweet()
    self.test_aspect_dict_list = get_test_aspect_dict_list(self.opinion_aspect_dict_list)
    self.display_ate_output()

  def display_ate_output(self):
    # Display aspect term extraction output on terminal
    display_ate_output_implement(self.original_test_text_list, 
                                self.preprocessed_test_text_list,
                                self.dependency_list, 
                                self.test_aspect_dict_list, 
                                self.extracted_aspect_list, 
                                self.opinion_aspect_dict_list)

  def remove_no_aspect_tweet(self):
    '''
        Remove tweets with no aspect to list before aspect term
        polarity classification
    '''
    (self.original_test_text_list,
        self.preprocessed_test_text_list,
        self.dependency_list,
        self.opinion_aspect_dict_list,
        self.extracted_aspect_list) = check_tweet(self.original_test_text_list,
                                                  self.preprocessed_test_text_list,
                                                  self.dependency_list,
                                                  self.opinion_aspect_dict_list,
                                                  self.extracted_aspect_list)

  def classify_aspect(self, polarity, mode):
    '''
      Classify aspect polarity in the mode
      polarity can be pos, neg, or neu
      mode can be implement or test

      Data structures
      ---------------
      Input Parameters:
        polarity  : STRING
        mode      : STRING
    '''
    classifier = Classifier(self.preprocessed_train_text_list, 
                            self.train_aspect_dict_list,
                            self.preprocessed_test_text_list, 
                            self.test_aspect_dict_list,
                            self.extracted_aspect_list, 
                            polarity=polarity, mode=mode)
    ensemble_prob_list = classifier.classify()
    return ensemble_prob_list

  def classify_aspect_pol(self):
    # Classify positive, negative, and neutral aspect polarity    
    self.pos_ensemble_prob_list = self.classify_aspect(polarity="pos",
                                                      mode="implement")
    self.neg_ensemble_prob_list = self.classify_aspect(polarity="neg",
                                                      mode="implement")
    self.neu_ensemble_prob_list = self.classify_aspect(polarity="neu",
                                                      mode="implement")

  def aggregate_aspect_pol(self):
    # Aggregate aspect polarities probability for aspect polarity
    self.aspect_pol_list = get_aspect_polarity(self.extracted_aspect_list,
                                              self.test_aspect_dict_list,
                                              self.pos_ensemble_prob_list,
                                              self.neg_ensemble_prob_list,
                                              self.neu_ensemble_prob_list)

  def get_tweet_pol(self):
    # Aggregate aspect polarities probability for tweet polarity
    self.senti_score_list = get_senti_scores(self.pos_ensemble_prob_list,
                                            self.neg_ensemble_prob_list,
                                            self.neu_ensemble_prob_list)
    self.absa_label_list = get_sentence_polarity(self.test_aspect_dict_list,
                                                self.senti_score_list)

  def get_main_table_dict(self):
    # Get the main display table for front end
    self.main_table_dict_list = display_output(self.pos_ensemble_prob_list,
                                              self.neg_ensemble_prob_list,
                                              self.neu_ensemble_prob_list,
                                              self.original_test_text_list,
                                              self.preprocessed_test_text_list,
                                              self.extracted_aspect_list,
                                              self.aspect_pol_list,
                                              self.senti_score_list,
                                              self.absa_label_list)

  def get_tweet_pol_table_dict(self):
    # Get the sentence level polarity table dictionary
    self.tweet_pol_table_dict = get_tweet_pol(self.original_test_text_list,
                                              self.absa_label_list)