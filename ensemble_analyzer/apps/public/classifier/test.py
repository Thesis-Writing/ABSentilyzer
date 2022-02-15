'''
  This module is intended for the main function used in testing
  the aspect based sentiment analyzer
'''

# Author            : Afrahly Afable
# Calling Sequence  : get_train_data()
#                     get_test_data()
#											main()
# Date Written      : October 28, 2021
# Date Revised      : December 27, 2021
# Purpose           : Main function to run the aspect based sentiment analyzer

import warnings
import pickle
import time
import os

from ensemble_analyzer.apps.public.classifier.scripts.utils import *
from ensemble_analyzer.apps.public.classifier.scripts.display import *
from ensemble_analyzer.apps.public.classifier.scripts.final_aspect_polarity import *
from ensemble_analyzer.apps.public.classifier.scripts.metrics import get_absa_performance
from ensemble_analyzer.apps.public.classifier.scripts.validation import check_no_aspect_test
from ensemble_analyzer.apps.public.classifier.scripts.aspect_polarity_classifier import Classifier
from ensemble_analyzer.apps.public.classifier.scripts.aspect_term_extraction import AspectTermExtraction

cwd = os.getcwd()

def get_train_data():
  '''
		This function returns the training data with annotations

		Data Structures
		---------------
		Input:
			None
		Output:
			train_text_list					: LIST
			train_aspect_dict_list	: LIST
			train_label							: LIST
  '''
  train_path = 'ensemble_analyzer/apps/public/data/annotated/7/a2/train.pkl'
  (original_train_text_list,train_aspect_dict_list,
    train_label_list) = get_list_from_data_frame(train_path)

  preprocessed_train_list_path = 'ensemble_analyzer/apps/public/data/annotated/7/preprocessed_train_list.pkl'
  preprocessed_train_text_list = pickle.loads(
    open(os.path.join(cwd, preprocessed_train_list_path), 'rb').read())

  return (original_train_text_list, preprocessed_train_text_list,
          train_aspect_dict_list, train_label_list)

def get_test_data():
  '''
		This function returns the test data with annotations

		Data Structures
		---------------
		Input:
			None
		Output:
			test_text_list							: LIST
			preprocessed_test_text_list	: LIST
			test_aspect_dict_list				: LIST
			test_label									: LIST
  '''
  test_path = 'ensemble_analyzer/apps/public/data/annotated/7/a2/test.pkl'
  (original_test_text_list,test_aspect_dict_list,
    test_label_list) = get_list_from_data_frame(test_path)

  preprocessed_test_list_path = 'ensemble_analyzer/apps/public/data/annotated/7/preprocessed_test_list.pkl'
  preprocessed_test_text_list = pickle.loads(
    open(os.path.join(cwd, preprocessed_test_list_path), 'rb').read())

  return (original_test_text_list, preprocessed_test_text_list,
          test_aspect_dict_list,test_label_list)

def aspect_term_extraction(original_train_text_list, original_test_text_list,
                          preprocessed_train_text_list,
                          preprocessed_test_text_list,
                          train_aspect_dict_list, test_aspect_dict_list):

  '''
    This function returns the performance of aspect term
    extraction module
    
    Data Structures
    ---------------
    Input:
      original_train_text_list     : LIST
      original_test_text_list      : LIST
      preprocessed_train_text_list : LIST
      preprocessed_test_text_list  : LIST
      train_aspect_dict_list       : LIST
      test_aspect_dict_list        : LIST
    Returns:
      ate_accuracy  : FLOAT
      ate_precision : FLOAT
      ate_recall    : FLOAT
      ate_f1        : FLOAT
  '''

  # Combine train and test for aspect term extraction evaluation
  ate_orig_text_list = [*original_train_text_list, *original_test_text_list]
  ate_prep_text_list = [*preprocessed_train_text_list,
                        *preprocessed_test_text_list]
  ate_dict_list = [*train_aspect_dict_list, *test_aspect_dict_list]
  ate_aspect_list = get_aspect_from_dict_list(ate_dict_list)

  # Perform aspect term extraction
  ate = AspectTermExtraction(ate_prep_text_list, mode="test")
  (extracted_aspect_list, opinion_aspect_dict_list,
      dependency_list) = ate.get_extracted_aspects()

  # Display aspect term extraction output in terminal
  display_ate_output(ate_orig_text_list, ate_prep_text_list,
                    dependency_list, ate_dict_list, 
                    extracted_aspect_list, opinion_aspect_dict_list)

  # Compute ATE performance
  (ate_accuracy, ate_precision,
      ate_recall, ate_f1) = ate.get_ate_performance(ate_aspect_list,
                                                extracted_aspect_list)

  return (ate_accuracy, ate_precision, ate_recall, ate_f1)

def main():
  '''
		This code block serves as the main function used to test the aspect based
		sentiment analyzer

		Data Structures
		---------------
		Input:
			None
		Output:
			None
  '''
  with warnings.catch_warnings():
    start = time.time()
    warnings.simplefilter("ignore")

    # Get train and test data
    (original_train_text_list, preprocessed_train_text_list,
        train_aspect_dict_list, train_label_list) = get_train_data()

    (original_test_text_list, preprocessed_test_text_list,
        test_aspect_dict_list, test_label_list) = get_test_data()

    # Get the train and test input aspects only from dictionary
    train_aspect_list = get_aspect_from_dict_list(train_aspect_dict_list)
    test_aspect_list = get_aspect_from_dict_list(test_aspect_dict_list)

    '''Perform aspect term extraction'''
    (ate_accuracy, ate_precision,
        ate_recall, ate_f1) = aspect_term_extraction(original_train_text_list,
                                                    original_test_text_list,
                                                    preprocessed_train_text_list,
                                                    preprocessed_test_text_list,
                                                    train_aspect_dict_list,
                                                    test_aspect_dict_list)

    '''Perform sentence level polarity classification without aspects'''
    sent_classifier = Classifier(train_text_list=preprocessed_train_text_list,
                                test_text_list=preprocessed_test_text_list,
                                train_sent_label_list=train_label_list,
                                test_sent_label_list=test_label_list)
    sent_classifier.classify_sentence()

    '''Perform aspect level polarity classification'''
    # Check if test input has aspects
    (preprocessed_test_text_list, test_aspect_dict_list,
        test_aspect_list) = check_no_aspect_test(preprocessed_test_text_list,
                                                test_aspect_dict_list,
                                                test_aspect_list)

    # Classify positive aspects
    pos_classifier = Classifier(preprocessed_train_text_list,
                                train_aspect_dict_list,
                                preprocessed_test_text_list,
                                test_aspect_dict_list,
                                test_aspect_list,
                                polarity="pos", mode="test")
    (pos_ensemble_prob_list, pos_atpc_eval_dict) = pos_classifier.classify()

    # Classify negative aspects
    neg_classifier = Classifier(preprocessed_train_text_list,
                                train_aspect_dict_list,
                                preprocessed_test_text_list,
                                test_aspect_dict_list,
                                test_aspect_list,
                                polarity="neg", mode="test")
    (neg_ensemble_prob_list, neg_atpc_eval_dict) = neg_classifier.classify()

    # Classify neutral aspects
    neu_classifier = Classifier(preprocessed_train_text_list,
                                train_aspect_dict_list,
                                preprocessed_test_text_list,
                                test_aspect_dict_list,
                                test_aspect_list,
                                polarity="neu", mode="test")
    (neu_ensemble_prob_list, neu_atpc_eval_dict) = neu_classifier.classify()

    # Aggregate aspect polarities
    aspect_polarity_list = get_aspect_polarity(test_aspect_list,
                                                pos_ensemble_prob_list,
                                                neg_ensemble_prob_list,
                                                neu_ensemble_prob_list)

    '''Final sentence sentiment with aspects (ABSA)'''
    # Aggregate aspect polarities probability
    senti_score_list = get_sentence_senti_scores(pos_ensemble_prob_list,
                                                neg_ensemble_prob_list,
                                                neu_ensemble_prob_list)
    absa_label_list = get_sentence_polarity(senti_score_list)
    (sent_accuracy, sent_precision,
        sent_recall, sent_f1) = get_absa_performance(test_label_list,
                                                    absa_label_list)

    main_table_dict_list = display_final_output(pos_ensemble_prob_list,
                                                neg_ensemble_prob_list,
                                                neu_ensemble_prob_list,
                                                original_test_text_list,
                                                preprocessed_test_text_list,
                                                test_aspect_list,
                                                aspect_polarity_list,
                                                senti_score_list,
                                                absa_label_list)

    ate_eval_dict = {'Rule-based ATE': [ate_accuracy,ate_precision,ate_recall,ate_f1]}

    sent_level_eval_dict = {'ABSentilyzer': [sent_accuracy,sent_precision,sent_recall,sent_f1]}

    end = time.time()
    time_linear_process = end-start
    if time_linear_process >= 60:
      time_linear_process = time_linear_process/60

    print("\nStarted: %fs; \nEnded: %fs; \nSentiLyzer Process Time:%fs\n"
          % (start, end, time_linear_process))

    return (main_table_dict_list, ate_eval_dict, pos_atpc_eval_dict,
            neg_atpc_eval_dict, neu_atpc_eval_dict, sent_level_eval_dict)

if __name__ == "__main__":
  main()
