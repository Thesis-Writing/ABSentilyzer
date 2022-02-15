'''
  This module is intended for the functions used in printing and
  computing of the performance evaluation and metrics that will be displayed
  in the terminal.
'''

# Authors           : Afrahly Afable
# Calling Sequence  : get_sl_metrics(y_true, y_pred, classifier=None)
#                     get_al_metrics(y_true, y_pred, average, mode=None, classifier=None)
#                     get_matrix(y_true, y_pred)
#                     compute_scores(tn, fp, fn, tp)
#                     get_absa_performance(test_label_list, absa_label_list)
# Date Written      : January 10, 2021
# Date Revised      : January 27, 2021
# Purpose           : To maintain the scoring functions for the performance evaluation of
#                     the model in one module

from ensemble_analyzer.apps.public.classifier.scripts.utils import *
from ensemble_analyzer.apps.public.classifier.scripts.validation import *
from ensemble_analyzer.apps.public.classifier.scripts.display import display_metrics

def confusion_matrix(true_label, pred_label):
  tp=0
  fp=0
  fn=0
  tn=0
  
  if true_label==pred_label==1:
    tp += 1
  elif true_label==0 and pred_label==1:
    fp += 1
  elif true_label==1 and pred_label==0:
    fn += 1
  elif true_label==pred_label==0:
    tn += 1
  
  return (tp,fp,fn,tn)

def get_matrix(y_true, y_pred):
  multilabel_confusion_matrix = []
  g_tp = 0
  g_fp = 0
  g_fn = 0
  g_tn = 0
  
  for i in range(len(y_true)):
    tp=0
    fp=0
    fn=0
    tn=0
    true_label = y_true[i]
    pred_label = y_pred[i]
    tn_fp = []
    fn_tp = []
    matrix = []
    
    tp,fp,fn,tn = confusion_matrix(true_label, pred_label)
    g_tp += tp
    g_fp += fp
    g_fn += fn
    g_tn += tn
    
    tn_fp.extend((tn,fp))
    fn_tp.extend((fn,tp))
    matrix.extend((tn_fp,fn_tp))
    multilabel_confusion_matrix.append(matrix)
  
  return (multilabel_confusion_matrix,g_tp,g_fp,g_fn,g_tn)

def compute_scores(tn, fp, fn, tp):
  try:
    accuracy = (tp + tn) / (tp + tn + fp + fn)
  except:
    accuracy = 0
  try:
    precision = tp / (tp + fp)
  except:
    precision = 0
  try:
    recall = tp / (tp + fn)
  except:
    recall = 0
  try:
    f1 = 2 * ((precision*recall) / (precision+recall))
  except:
    f1 = 0
  
  return (accuracy, precision, recall, f1)

def get_multilabel_scores(multilabel_confusion_matrix):
  
  label_accuracy_score_list = []
  label_precision_score_list = []
  label_recall_score_list = []
  label_f1_score_list = []
  label_support_score_list = []

  for i in range(len(multilabel_confusion_matrix)):
    label_cm = multilabel_confusion_matrix[i]
    support = 0
    TN_FP = label_cm[0]
    FN_TP = label_cm[1]
    TN = TN_FP[0]
    FP = TN_FP[1]
    FN = FN_TP[0]
    TP = FN_TP[1]
    
    (accuracy, precision, recall, f1) = compute_scores(TN,FP,FN,TP)
    
    label_accuracy_score_list.append(accuracy)
    label_precision_score_list.append(precision)
    label_recall_score_list.append(recall)
    label_f1_score_list.append(f1)
    
    if precision != 0 or recall != 0 or f1 != 0:
      support += 1
    label_support_score_list.append(support)
    
  return (label_accuracy_score_list,
          label_precision_score_list,
          label_recall_score_list,
          label_f1_score_list,
          label_support_score_list)

def get_al_metrics(y_true, y_pred, average, mode=None, classifier=None):
  total_accuracy = 0
  total_precision = 0
  total_recall = 0
  total_f1 = 0
  
  tp = 0
  fp = 0
  tn = 0
  fn = 0

  if mode=="model":
    f = open("al_samples.txt", "a")
    g = open("al_output.txt", "a")

    if classifier:
      f.write("\n{}".format(classifier))
      g.write("\n{}".format(classifier))
    f.write("\n======================================================================\n")
    g.write("\n======================================================================\n")

  for i in range(len(y_true)):
    true = check_instance(y_true[i])
    pred = check_instance(y_pred[i])

    (multilabel_confusion_matrix,
        g_tp,g_fp,g_fn,g_tn) = get_matrix(true,pred)

    if is_equal_zero(true,pred):
      (temp_accuracy, temp_precision, temp_recall, temp_f1) = (1,1,1,1)
      tp += 1
      fp += 1
      tn += 1
      fn += 1
    else:
      (temp_accuracy, temp_precision, temp_recall, temp_f1) = compute_scores(g_tn,g_fp,g_fn,g_tp)
      tp += g_tp
      fp += g_fp
      tn += g_tn
      fn += g_fn
    if mode=="model":
      f.write("\nSample: {}".format(i+1))
      f.write("  \nTrue:  {}   Pred: {}".format(true,pred))
      f.write("  \nAccuracy:  {:.4f}".format(temp_accuracy))
      f.write("  \nPrecision: {:.4f}".format(temp_precision))
      f.write("  \nRecall:    {:.4f}".format(temp_recall))
      f.write("  \nF1:        {:.4f}\n".format(temp_f1))

    total_accuracy += temp_accuracy
    total_precision += temp_precision
    total_recall += temp_recall
    total_f1 += temp_f1

  avg_accuracy = total_accuracy / len(y_true)
  avg_precision = total_precision / len(y_true)
  avg_recall = total_recall / len(y_true)
  avg_f1 = total_f1 / len(y_true)

  if mode=="model":
    g.write("\n{}".format(average.upper()))
    g.write("\nTotal Accuracy:  {}".format(total_accuracy))
    g.write("\nTotal Precision: {}".format(total_precision))
    g.write("\nTotal Recall:    {}".format(total_recall))
    g.write("\nTotal F1:        {}\n".format(total_f1))

    g.write("\nAVG-Accuracy:  {}".format(avg_accuracy))
    g.write("\nAVG-Precision: {}".format(avg_precision))
    g.write("\nAVG-Recall:    {}".format(avg_recall))
    g.write("\nAVG-F1:        {}\n".format(avg_f1))

    g.write("\nTP: {}".format(tp))
    g.write("\nFP: {}".format(fp))
    g.write("\nTN: {}".format(tn))
    g.write("\nFN: {}".format(fn))

    f.write("\n======================================================================\n")
    g.write("\n======================================================================\n")
    f.close()
    g.close()

  return (avg_accuracy,avg_precision,avg_recall,avg_f1)

def get_sl_metrics(y_true, y_pred, classifier=None):
  from sklearn.metrics import accuracy_score, precision_score
  from sklearn.metrics import recall_score, f1_score, confusion_matrix
  import numpy as np
  import pandas as pd

  f = open("sl_samples.txt", "a")
  g = open("sl_output.txt", "a")

  if classifier:
    f.write("\n{}".format(classifier))
    g.write("\n{}".format(classifier))
  f.write("\n======================================================================\n")
  for i in range(len(y_true)):
    f.write("\nTrue:  {}   Pred: {}".format(y_true[i],y_pred[i]))
  f.write("\n======================================================================\n")

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
  recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
  f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
  cnf_matrix = confusion_matrix(y_true, y_pred)

  fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
  fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
  tp = np.diag(cnf_matrix)
  tn = cnf_matrix.sum() - (fp + fn + tp)

  g.write("\n======================================================================\n")
  g.write("\nAccuracy:  {}".format(accuracy))
  g.write("\nPrecision: {}".format(precision))
  g.write("\nRecall:    {}".format(recall))
  g.write("\nF1:        {}".format(f1))
  g.write("\nTP: {}".format(tp))
  g.write("\nFP: {}".format(fp))
  g.write("\nTN: {}".format(tn))
  g.write("\nFN: {}".format(fn))
  g.write("\nConfusion Matrix:\n")
  g.write("{}\n".format(cnf_matrix))
  g.write("\n======================================================================\n")

  return (accuracy, precision, recall, f1, tp, fp, tn, fn)

def get_absa_performance(test_label_list, absa_label_list):
  print("==============================================================")
  print("             SENTENCE LEVEL WITH ASPECT EVALUATION            ")
  print("==============================================================")

  # Encode labels to numerical representation
  bin_true = get_sentence_data_frame(label_list=test_label_list)
  bin_absa = get_sentence_data_frame(label_list=absa_label_list)
  
  bin_true = bin_true.values.tolist()
  bin_absa = bin_absa.values.tolist()
  
  bin_true = encode_label(bin_true)
  bin_absa = encode_label(bin_absa)
  
  (accuracy, precision,
      recall, f1, tp, fp, tn, fn) = get_sl_metrics(bin_true, bin_absa)
  display_metrics(accuracy, precision, recall, f1)
  print("ABSA-TP: {}, ABSA-FP: {}, ABSA-TN: {}, ABSA-FN: {}".format(tp, fp,
                                                                    tn, fn))

  return (accuracy, precision, recall, f1)


