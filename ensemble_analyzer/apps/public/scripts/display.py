'''
  This module is intended for the functions that display output in the terminal
'''

# Author            : Afrahly Afable
# Calling Sequence  : display_initial_output(original_input_list,preprocessed_input_list,extracted_aspect_list,opinion_aspect_dict_list,dependency_list,pre_labelled_aspects)
#                     display_output(pos_ensemble_prob_list,neg_ensemble_prob_list,neu_ensemble_prob_list,original_input_list,preprocessed_input_list,extracted_aspect_list,final_aspect_polarity_list,final_polarity_list)
#                     display_header(mode)
#                     display_footer(mode)
#                     display_metrics(final_accuracy,final_precision,final_recall,final_f1)
#                     display_model_performance
# Date Written      : December 1, 2021
# Date Revised      : December 27, 2021
# Purpose           : Display output in the terminal

import pandas as pd

def display_initial_output(original_input_list, preprocessed_input_list, 
                          extracted_aspect_list, opinion_aspect_dict_list, 
                          dependency_list,pre_labelled_aspects):
  '''
    This function displays the outputs of preprocessing and aspect term extraction
    
    Data Structure
    --------------
    Input:
      original_input_list       : LIST
      preprocessed_input_list   : LIST
      extracted_aspect_list     : LIST
      opinion_aspect_dict_list  : LIST
      dependency_list           : LIST 
      pre_labelled_aspects      : LIST
    Returns:
      None
  ''' 
  
  print("==============================================================")
  print("                   DISPLAY DATA BEFORE ATPC                   ")
  print("==============================================================")
  
  for i in range(len(original_input_list)):
    original_input = original_input_list[i]
    preprocessed_input = preprocessed_input_list[i]
    input_aspects = extracted_aspect_list[i]
    opinion_aspect = opinion_aspect_dict_list[i]
    dependency = dependency_list[i]
    pre_labelled = pre_labelled_aspects[i]
    
    print("Input {}:".format(i+1))
    print(" | ORIGINAL TEXT:")
    print(" | {}".format(original_input))
    print(" | PREPROCESSED TEXT:")
    print(" | {}".format(preprocessed_input))
    print(" | INPUT ASPECTS:")
    print(" | {}".format(input_aspects))
    print(" | OPINION-ASPECT:")
    print(" | {}".format(opinion_aspect))
    print(" | DEPENDENCIES:")
    print(" | {}".format(dependency))
    print(" | PRE-LABELLED ASPECTS:")
    print(" | {}".format(pre_labelled))
    print("\n")
    
  print("==============================================================")

def display_output(pos_ensemble_prob_list, neg_ensemble_prob_list,
                        neu_ensemble_prob_list, original_input_list,
                        preprocessed_input_list, extracted_aspect_list,
                        aspect_polarity_list, senti_score_list, 
                        absa_label_list):
  '''
    This function display the output of preprocessing, aspect term extraction, 
    and aspect term polarity classification then returns a list of 
    dictionary containing the outputs
    
    Data Structures
    ---------------
    Input:
      pos_ensemble_prob_list     : LIST
      neg_ensemble_prob_list     : LIST
      neu_ensemble_prob_list     : LIST
      original_input_list        : LIST
      preprocessed_input_list    : LIST
      extracted_aspect_list      : LIST
      aspect_polarity_list : LIST
      final_polarity_list        : LIST
      senti_score_list           :
      final_polarity_list_2      :
    Returns:
      main_table_dict_list       : LIST
  '''

  print("\n")
  print("==============================================================")
  print("                        FINAL OUTPUT                          ")
  print("==============================================================")
  
  main_table_dict_list = {}

  for i in range(len(preprocessed_input_list)):
    table_row = []
    
    pos_proba = pos_ensemble_prob_list[i]
    neg_proba = neg_ensemble_prob_list[i]
    neu_proba = neu_ensemble_prob_list[i]
    
    original_input = original_input_list[i]
    preprocessed_input = preprocessed_input_list[i]
    input_aspects = extracted_aspect_list[i]
    aspect_polarities = aspect_polarity_list[i]
    senti_score = senti_score_list[i]
    absa_polarity = absa_label_list[i]
    
    print("Input {}:".format(i+1))
    
    print(" | ORIGINAL TEXT:".format(original_input))
    print(" | {}".format(original_input))
    
    print(" | PREPROCESSED TEXT:".format(preprocessed_input))
    print(" | {}".format(preprocessed_input))
    
    print(" | INPUT ASPECTS:")
    print(" | {}".format(input_aspects))
    
    print(" | POS ASPECT PROBA:")
    print(" | {}".format(pos_proba))
    
    print(" | NEG ASPECT PROBA:")
    print(" | {}".format(neg_proba))
    
    print(" | NEU ASPECT PROBA:")
    print(" | {}".format(neu_proba))
    
    print(" | SENTI SCORE:")
    print(" | {}".format(senti_score))
    
    print(" | ASPECT POLARITIES:")
    print(" | {}".format(aspect_polarities))
    
    print(" | AB-SENTENCE POLARITY:")
    print(" | {}".format(absa_polarity))
    
    print("\n")
    
    # table_row['Original Text'] = original_input
    # table_row['Preprocessed Text'] = preprocessed_input
    # table_row['Aspect Polarities'] = aspect_polarities
    # table_row['AB-Sentence Polarity'] = absa_polarity
    # main_table_dict_list.append(table_row)
    
    table_row.append(original_input)
    table_row.append(input_aspects)
    table_row.append(aspect_polarities)
    main_table_dict_list[i+1] = table_row
    
  print("==============================================================")
  
  return main_table_dict_list

def display_header(polarity):
  '''
    Simple function that displays header
    
    Data Structures
    ---------------
    Input:
      polarity  : STRING
    Output:
      None
  '''
  if polarity == "pos":
    print("==============================================================")
    print("             POSITIVE ASPECT LEVEL CLASSIFICATION             ")
    print("==============================================================")
  elif polarity == "neg":
    print("==============================================================")
    print("             NEGATIVE ASPECT LEVEL CLASSIFICATION             ")
    print("==============================================================")
  elif polarity == "neu":
    print("==============================================================")
    print("             NEUTRAL ASPECT LEVEL CLASSIFICATION              ")
    print("==============================================================")

def display_footer(polarity):
  '''
    Simple function that displays footer
    
    Data Structures
    ---------------
    Input:
      polarity  : STRING
    Output:
      None
  '''
  
  if polarity == "pos":
    print("==============================================================")
    print("       POSITIVE ASPECT LEVEL CLASSIFICATION EVALUATION        ")
    print("==============================================================")
  elif polarity == "neg":
    print("==============================================================")
    print("       NEGATIVE ASPECT LEVEL CLASSIFICATION EVALUATION        ")
    print("==============================================================")
  elif polarity == "neu":
    print("==============================================================")
    print("       NEUTRAL ASPECT LEVEL CLASSIFICATION EVALUATION         ")
    print("==============================================================")

def display_metrics(accuracy, precision, recall, f1):
  '''
    This function prints the metrics based on final score.
    
    Data Structures
    ---------------
    Input:
      - accuracy  : FLOAT
      - precision : FLOAT
      - recall    : FLOAT
      - f1        : FLOAT
    Returns:
      - None
  '''
  print("Accuracy:  {:.4f}".format(accuracy))
  print("Precision: {:.4f}".format(precision))
  print("Recall:    {:.4f}".format(recall))
  print("F1-score:  {:.4f}".format(f1))

def display_model_performance(svm_accuracy, svm_precision, svm_recall, svm_f1, 
                              mnb_accuracy, mnb_precision, mnb_recall, mnb_f1,
                              ensemble_accuracy, ensemble_precision, 
                              ensemble_recall, ensemble_f1):
  '''
    This function prints the performance evaluation of the model based 
    on their final score.
    
    Data Structures
    ---------------
    Input:
      - svm_accuracy        : FLOAT
      - svm_precision       : FLOAT
      - svm_recall          : FLOAT
      - svm_f1              : FLOAT    
      - mnb_accuracy        : FLOAT
      - mnb_precision       : FLOAT
      - mnb_recall          : FLOAT
      - mnb_f1              : FLOAT
      - ensemble_accuracy   : FLOAT
      - ensemble_precision  : FLOAT  
      - ensemble_recall     : FLOAT
      - ensemble_f1         : FLOAT
    Returns:
      - None
  '''
  print("\nSVM PERFORMANCE EVALUATION:")
  display_metrics(svm_accuracy, svm_precision, svm_recall, svm_f1)
  print("\nMNB PERFORMANCE EVALUATION:")
  display_metrics(mnb_accuracy, mnb_precision, mnb_recall, mnb_f1)
  print("\nENSEMBLE PERFORMANCE EVALUATION:")
  display_metrics(ensemble_accuracy, ensemble_precision, 
                  ensemble_recall, ensemble_f1)

def format_display(t):
  import re
  import textwrap
  t=re.sub('\s+',' ',t); t=re.sub('^\s+','',t); t=re.sub('\s+$','',t)
  t=textwrap.wrap(t,width=100,initial_indent=' '*1,subsequent_indent=' '*15)
  s=""
  for i in (t): s=s+i+"\n"
  s=re.sub('\s+$','',s)
  return(s)

def save_confusion_matrix(tp, fp, fn, tn, model=str, polarity=None):
  df = pd.DataFrame(list(zip(tp, fp, fn, tn)), 
                    columns = ["TP", "FP", "FN", "TN"])
  if polarity != None:
    df.to_csv("{}_{}_al_cm.csv".format(model,polarity))
  elif model == 'ate':
    df.to_csv("{}_cm.csv".format(model))

def display_ate_output(ate_orig_text_list, ate_prep_text_list,
                      dependency_list, ate_dict_list, 
                      extracted_aspect_list, opinion_aspect_dict_list):

  for i in range(len(extracted_aspect_list)):
    print(i+1)
    print(f"Original:     {format_display(str(ate_orig_text_list[i]))}")
    print(f"Preprocessed: {format_display(str(ate_prep_text_list[i]))}")
    print(f"Dependency:   {format_display(str(dependency_list[i]))}")
    print(f"Annotated:    {format_display(str(ate_dict_list[i]))}")
    print(f"Extracted:    {format_display(str(extracted_aspect_list[i]))}")
    print(f"Opinion-Dict: {format_display(str(opinion_aspect_dict_list[i]))}")
    print("\n")

def display_ate_output_implement(ate_orig_text_list, ate_prep_text_list,
                      dependency_list, ate_dict_list, 
                      extracted_aspect_list, opinion_aspect_dict_list):

  for i in range(len(extracted_aspect_list)):
    print(i+1)
    print(f"Original:     {format_display(str(ate_orig_text_list[i]))}")
    print(f"Preprocessed: {format_display(str(ate_prep_text_list[i]))}")
    print(f"Dependency:   {format_display(str(dependency_list[i]))}")
    print(f"Pre-label:    {format_display(str(ate_dict_list[i]))}")
    print(f"Extracted:    {format_display(str(extracted_aspect_list[i]))}")
    print(f"Opinion-Dict: {format_display(str(opinion_aspect_dict_list[i]))}")
    print("\n")

def get_prep_table(original_input_list,preprocessed_input_list):
  preprocessed_table_dict = {}
    
  for i in range(len(preprocessed_input_list)):
    table_row = []
    
    original_input = original_input_list[i]
    preprocessed_input = preprocessed_input_list[i]

    table_row.append(original_input)
    table_row.append(preprocessed_input)
    preprocessed_table_dict[i+1] = table_row
      
  return preprocessed_table_dict

def get_tweet_pol(original_input_list,final_polarity_list):
  final_sentence_polarity_table_dict = {}
    
  for i in range(len(original_input_list)):
    table_row = []
    
    original_input = original_input_list[i]
    final_sentence_polarity = final_polarity_list[i]

    table_row.append(original_input)
    table_row.append(final_sentence_polarity)
    final_sentence_polarity_table_dict[i+1] = table_row
      
  return final_sentence_polarity_table_dict