# Program Title         : ensemble.py
# Author                : Afrahly Afable
# General System Design : This module is intended for the ensemble algorithm 
#                         functions to ensemble the results of SVM and MNB 
#                         after the aspect term polarity classification,
#                         the algorithm was based on the study conducted by 
#                         Korovkinas et. al. (2017) but was coded from scratch
#                         for implementation.
# Date Written          : October 28, 2021
# Date Revised          : December 27, 2021
# Purpose               : Ensemble the results of SVM and MNB algorithms
# Data Structures       : List, Dictionary, Integer

import numpy as np
import math

def combine_final_prediction_to_array(similar_predictions, 
                                      ensemble_predictions, 
                                      ensemble_prob):
  similar_prediction = {}
  final_prediction = {}

  # Combine similar and ensemble pred
  for i in range(len(similar_predictions)):
    similar_results_dict = similar_predictions[i]
    for row_num,row_data in similar_results_dict.items():
      prediction = row_data["svm_pred"]
      similar_prediction[row_num] = prediction
      
  for i in range(len(ensemble_predictions)):
    final_results_dict = ensemble_predictions[i]
    for key in final_results_dict:
      final_prediction[key] = final_results_dict.get(key)
      
  similar_prob = {}
  final_prob = {}

  # Combine similar and ensemble prob
  for i in range(len(similar_predictions)):
    similar_results_dict = similar_predictions[i]
    for row_num,row_data in similar_results_dict.items():
      prob = row_data["svm_prob"]
      similar_prob[row_num] = prob
  
  for i in range(len(ensemble_prob)):
    final_results_dict = ensemble_prob[i]
    for key in final_results_dict:
      final_prob[key] = final_results_dict.get(key)
      
  similar_prediction.update(final_prediction)
  similar_prob.update(final_prob)

  final_combined_pred_dict = {k: v for k, v in sorted(similar_prediction.items())}
  final_combined_prob_dict = {k: v for k, v in sorted(similar_prob.items())}

  final_combined_pred_dict_list = []
  final_combined_prob_dict_list = []

  for key in final_combined_pred_dict:
    prediction = final_combined_pred_dict.get(key)
    final_combined_pred_dict_list.append(prediction)
    
  for key in final_combined_prob_dict:
    prob = final_combined_prob_dict.get(key)
    final_combined_prob_dict_list.append(prob)
  
  final_prediction_array = np.array(final_combined_pred_dict_list)
  final_prob_array = np.array(final_combined_prob_dict_list)

  
  return (final_prediction_array, final_prob_array)

def ensemble_algorithm(different_predictions):
  
  combined_pred_dict_list = []
  combined_prob_dict_list = []
  
  for i in range(len(different_predictions)):
    row_dict = different_predictions[i]
    for row_num,row_data in row_dict.items():
      for key in row_data:
        svm_pred = row_data["svm_pred"]
        mnb_pred = row_data["mnb_pred"]
        svm_prob = row_data["svm_prob"]
        mnb_prob = row_data["mnb_prob"]
        
        combined_pred = []
        combined_prob = []
        
        if isinstance(svm_pred, list):
          if len(svm_pred) == len(mnb_pred):
            for i in range(len(svm_pred)):
              svm_aspect_prob = svm_prob[i]
              mnb_aspect_prob = mnb_prob[i]
              
              difference = 0
              average = 0
              
              if mnb_aspect_prob == 0:
                difference += svm_aspect_prob
              elif svm_aspect_prob == 0:
                difference += math.log(mnb_aspect_prob,10)
              else:
                difference += svm_aspect_prob + math.log(mnb_aspect_prob,10)
                average = difference/2
                
              if difference <= average:
                combined_pred.append(svm_pred[i])
                combined_prob.append(svm_aspect_prob)
              else:
                combined_pred.append(mnb_pred[i])
                combined_prob.append(mnb_aspect_prob)
        elif isinstance(svm_pred, int):
          svm_aspect_prob = svm_prob[1]
          mnb_aspect_prob = mnb_prob[1]
          
          difference = 0
          average = 0
          
          if mnb_aspect_prob == 0:
            difference += svm_aspect_prob
          elif svm_aspect_prob == 0:
            difference += math.log(mnb_aspect_prob,10)
          else:
            difference += svm_aspect_prob + math.log(mnb_aspect_prob,10)
            average = difference/2
            
          if difference <= average:
            combined_pred.append(svm_pred)
            combined_prob.extend(svm_prob)
          else:
            combined_pred.append(mnb_pred)
            combined_prob.extend(mnb_prob)
      
      combined_pred_dict_list.append({row_num: combined_pred})
      combined_prob_dict_list.append({row_num: combined_prob})

  return (combined_pred_dict_list, combined_prob_dict_list)

def ensemble(svm_predictions,mnb_predictions,svm_score,mnb_probabilities):

  svm_pred_list = svm_predictions.tolist()
  mnb_pred_list = mnb_predictions.tolist()
  svm_prob_list = svm_score.tolist()
  mnb_prob_list = mnb_probabilities.tolist()
  
  if len(svm_pred_list) == len(mnb_pred_list):
    similar_results_dict_list = []
    diff_results_dict_list = []
    for row in range(len(svm_pred_list)):
      if svm_pred_list[row] == mnb_pred_list[row]:
        # 1st Find results which are the same in both SVM and Naive Bayes
        similar_results_dict_list.append({row:{"svm_pred": svm_pred_list[row], 
                                              "mnb_pred":mnb_pred_list[row], 
                                              "svm_prob": svm_prob_list[row], 
                                              "mnb_prob": mnb_prob_list[row]}})
      else:
        # 2nd Find results which are different between SVM and Na¨ıve Bayes classification.
        diff_results_dict_list.append({row:{"svm_pred": svm_pred_list[row], 
                                            "mnb_pred":mnb_pred_list[row], 
                                            "svm_prob": svm_prob_list[row], 
                                            "mnb_prob": mnb_prob_list[row]}})
    
    (ensemble_predictions,
        ensemble_prob) = ensemble_algorithm(diff_results_dict_list)
    (final_pred_array,
        final_prob_array) = combine_final_prediction_to_array(similar_results_dict_list,
                                                              ensemble_predictions,
                                                              ensemble_prob)

  return (final_pred_array, final_prob_array)