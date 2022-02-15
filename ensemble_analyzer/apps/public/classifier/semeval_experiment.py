'''
  This script is intended for the main function used in evaluating
  the aspect term extraction rules on SemEval 2014 laptops and
  restaurants dataset
'''

# Author            : Afrahly Afable
# Date Written      : October 28, 2021
# Date Revised      : December 27, 2021
# Purpose           : Evaluate ATE rules using SemEval 2014 dataset

import pickle
import os

from ensemble_analyzer.apps.public.classifier.scripts.utils import *
from ensemble_analyzer.apps.public.classifier.scripts.display import *
from ensemble_analyzer.apps.public.classifier.scripts.display import *
from ensemble_analyzer.apps.public.classifier.scripts.final_aspect_polarity import *
from ensemble_analyzer.apps.public.classifier.scripts.aspect_term_extraction import AspectTermExtraction

cwd = os.getcwd()

def get_list_from_data_frame(data_path):
  data_frame = pickle.loads(open(os.path.join(cwd, data_path), 'rb').read())

  text_list = data_frame['text'].tolist()
  preprocessed_text_list = data_frame['preprocessed_text'].tolist()
  aspect_list = data_frame['aspectTerms'].tolist()

  return text_list,preprocessed_text_list,aspect_list


def display_output(aspect_list, extracted_aspect, opinion_aspect_dict_list, 
                  dependency_list, text_list):
  for i in range(len(aspect_list)):
    print(i)
    print("Text:         {}".format(text_list[i]))
    print("Annotated:    {}".format(aspect_list[i]))
    print("Extracted:    {}".format(extracted_aspect[i]))
    print("Opinion Dict: {}".format(opinion_aspect_dict_list[i]))
    print("Dependency:   {}\n".format(dependency_list[i]))

def main():
  laptops_path = r'ensemble_analyzer/apps/public/data/experiment/sem_eval_laptops_restaurant/laptops.pkl'
  (laptops_text_list, laptops_preprocessed_text_list, 
      laptops_aspect_list) = get_list_from_data_frame(laptops_path)

  restaurants_path = r'ensemble_analyzer/apps/public/data/experiment/sem_eval_laptops_restaurant/restaurants.pkl'
  (restaurants_text_list, restaurants_preprocessed_text_list, 
      restaurants_aspect_list) = get_list_from_data_frame(restaurants_path)

  # laptops_ate = AspectTermExtraction(laptops_text_list, mode="test")
  # (laptops_extracted_aspect_list, laptops_opinion_aspect_dict_list, 
  #     laptops_dependency_list) = laptops_ate.get_extracted_aspects()

  # display_ate_output(laptops_text_list, laptops_preprocessed_text_list,
  #                   laptops_dependency_list, laptops_aspect_list, 
  #                   laptops_extracted_aspect_list,
  #                   laptops_opinion_aspect_dict_list)

  # print("=======================================================")
  # print("                       LAPTOPS                         ")
  # print("=======================================================")

  # # compute_performance(laptops_aspect_list, laptops_extracted_aspect_list)
  # (ate_accuracy, ate_precision,
  #     ate_recall, ate_f1) = laptops_ate.get_ate_performance(laptops_aspect_list,
  #                                               laptops_extracted_aspect_list)

  # Perform aspect term extraction
  restaurants_ate = AspectTermExtraction(restaurants_text_list, 
                                        mode="test")
  (restaurants_extracted_aspect_list, restaurants_opinion_aspect_dict_list, 
      restaurants_dependency_list) = restaurants_ate.get_extracted_aspects()

  display_ate_output(restaurants_text_list, restaurants_preprocessed_text_list,
                    restaurants_dependency_list, restaurants_aspect_list, 
                    restaurants_extracted_aspect_list,
                    restaurants_opinion_aspect_dict_list)

  print("=======================================================")
  print("                     RESTAURANTS                       ")
  print("=======================================================")
  # compute_performance(restaurants_aspect_list, restaurants_extracted_aspect_list)
  (ate_accuracy, ate_precision,
      ate_recall, ate_f1) = restaurants_ate.get_ate_performance(restaurants_aspect_list,
                                                restaurants_extracted_aspect_list)

if __name__ == "__main__":
  main()