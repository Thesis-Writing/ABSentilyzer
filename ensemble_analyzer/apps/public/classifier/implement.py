'''
  This module is intended for the main function used in testing 
  the aspect based sentiment analyzer
'''

# Author            : Afrahly Afable
# Calling Sequence  : get_train_data()
#                     aspect_term_extraction()
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
  train_path = 'ensemble_analyzer/apps/public/data/annotated/7/a1/train.pkl'
  (original_train_text_list,train_aspect_dict_list,
    train_label_list) = get_list_from_data_frame(train_path)

  preprocessed_train_list_path = 'ensemble_analyzer/apps/public/data/annotated/7/preprocessed_train_list.pkl'
  preprocessed_train_text_list = pickle.loads(
    open(os.path.join(cwd, preprocessed_train_list_path), 'rb').read())

  return (original_train_text_list, preprocessed_train_text_list, 
          train_aspect_dict_list, train_label_list)

def aspect_term_extraction(original_test_text_list,
                          preprocessed_test_text_list,
                          test_aspect_dict_list):
  '''
    This function returns the output of aspect term
    extraction module
    
    Data Structures
    ---------------
    Input:
      original_test_text_list     : LIST
      preprocessed_test_text_list : LIST
      test_aspect_dict_list       : LIST
    Returns:
      extracted_aspect_list    : LIST 
      opinion_aspect_dict_list : LIST   
      dependency_list          : LIST 
      test_aspect_dict_list    : LIST 
  '''

  # Perform aspect term extraction
  ate = AspectTermExtraction(preprocessed_test_text_list, mode="implement")
  (extracted_aspect_list, opinion_aspect_dict_list, 
      dependency_list, test_aspect_dict_list) = ate.get_extracted_aspects()

  # Display aspect term extraction output in terminal
  display_ate_output_implement(original_test_text_list, 
                              preprocessed_test_text_list,
                              dependency_list, 
                              test_aspect_dict_list, 
                              extracted_aspect_list, 
                              opinion_aspect_dict_list)

  return (extracted_aspect_list, opinion_aspect_dict_list, 
        dependency_list, test_aspect_dict_list)


def main(user_input=None):
  '''
		This code block serves as the main function used to test the aspect based 
		sentiment analyzer
  
		Data Structures
		---------------
		Input:
			user_input  : STRING or LIST or Nnone
		Output:
			None	
  '''
  with warnings.catch_warnings():
    start = time.time()
    warnings.simplefilter("ignore")

    # Get train and test data
    (original_train_text_list, preprocessed_train_text_list,
        train_aspect_dict_list, train_label_list) = get_train_data()

    (original_test_text_list, 
        preprocessed_test_text_list) = get_user_input(user_input)

    # Perform aspect term extraction
    test_aspect_dict_list = []
    (extracted_aspect_list, 
        opinion_aspect_dict_list, 
        dependency_list, 
        test_aspect_dict_list) = aspect_term_extraction(original_test_text_list,
                                                        preprocessed_test_text_list,
                                                        test_aspect_dict_list)

    # Get the train and test input aspects only from dictionary
    train_aspect_list = get_aspect_from_dict_list(train_aspect_dict_list) 
    test_aspect_list = get_aspect_from_dict_list(test_aspect_dict_list)

    # Check if test input has aspects
    (original_test_text_list,
        preprocessed_test_text_list,
        test_aspect_dict_list,
        test_aspect_list) = check_no_aspect_test(original_test_text_list,
                                                preprocessed_test_text_list,
                                                test_aspect_dict_list,
                                                test_aspect_list)

    # Classify positive aspects
    pos_classifier = Classifier(preprocessed_train_text_list, 
                                train_aspect_dict_list,
                                preprocessed_test_text_list, 
                                test_aspect_dict_list,
                                test_aspect_list, 
                                polarity="pos", mode="implement")
    pos_ensemble_prob_list = pos_classifier.classify()

    # Classify negative aspects
    neg_classifier = Classifier(preprocessed_train_text_list, 
                                train_aspect_dict_list,
                                preprocessed_test_text_list, 
                                test_aspect_dict_list,
                                test_aspect_list, 
                                polarity="neg", mode="implement")
    neg_ensemble_prob_list = neg_classifier.classify()

    # Classify neutral aspects
    neu_classifier = Classifier(preprocessed_train_text_list, 
                                train_aspect_dict_list,
                                preprocessed_test_text_list, 
                                test_aspect_dict_list,
                                test_aspect_list, 
                                polarity="neu", mode="implement")
    neu_ensemble_prob_list = neu_classifier.classify()

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
    
    main_table_dict_list = display_final_output(pos_ensemble_prob_list,
                                                neg_ensemble_prob_list,
                                                neu_ensemble_prob_list,
                                                original_test_text_list,
                                                preprocessed_test_text_list,
                                                test_aspect_list,
                                                aspect_polarity_list,
                                                senti_score_list,
                                                absa_label_list)
    preprocessed_table_dict = get_preprocessed_table(original_test_text_list,
                                                    preprocessed_test_text_list)
    final_sentence_polarity_table_dict = get_final_sentence_polarity_table(original_test_text_list,
                                                                          absa_label_list)

    end = time.time()
    time_linear_process = end-start

    print("\nStarted: %fs; \nEnded: %fs; \nSentiLyzer Process Time:%fs\n"
          % (start, end, time_linear_process))
    return (main_table_dict_list, 
            preprocessed_table_dict, 
            final_sentence_polarity_table_dict)

# if __name__ == "__main__":
#   user_input = ["I really love the pizza!"]
#   main(user_input)