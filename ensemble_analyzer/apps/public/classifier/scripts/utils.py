# Author            : Afrahly Afable
# Date Written      : November 10, 2021
# Date Revised      : December 27, 2021
# Purpose           : To maintain utility functions used in different modules

import os
import pickle
import numpy as np
import pandas as pd
from ensemble_analyzer.apps.public.classifier.scripts import preprocessing

cwd = os.getcwd()


def get_list_from_data_frame(data_path):
  '''
    Takes the path of sentence level data frame containing annotations and 
    returns the texts as list, the aspects with annotated labels, 
    and the sentence level annotation
    
    Data Structures
    ---------------
    Input:
      data_path            : STRING
    Returns:
      tweet_list           : LIST
      aspect_list          : LIST
      sentence_label_list  : LIST
  '''
  
  data_frame = pickle.loads(open(os.path.join(cwd, data_path), 'rb').read())
    
  tweet_list = data_frame['tweets'].tolist()
  aspect_list = data_frame['annotated_aspects'].tolist()
  sentence_label_list = data_frame['sentence_label'].tolist()
  
  return tweet_list,aspect_list,sentence_label_list

def get_all_aspects(aspect_list, is_list=False):
  '''
    Return all aspects in the given aspect list
    if list is True, the aspect list is in a list format 
    (Example: [[aspect1, aspect2]])
    if list is False, the aspect list contains list of {aspect:polarity} 
    (Example: [[{aspect1:polarity}, {aspect2:polarity}]])
    
    Data Structures
    ---------------
    Input:
      aspect_list          : LIST
      list                 : STRING
    Returns:
      aspects              : LIST
  '''
  aspects=[]
  
  if is_list == True:
    for inner_list in aspect_list:
      for aspect in inner_list:
        aspects.append(aspect)
  else:
    for inner_list in aspect_list:
      for _dict in inner_list:
        for key in _dict:
          aspects.append(key)

  # aspects = list(dict.fromkeys(aspects))

  return aspects

def get_data_frame(text_list, aspect_dict_list, target_aspect_list):
  '''
    Accepts the list of text, aspect, and the extracted aspect as input.
    Returns data frame with tweets as rows and aspects in 
    the extracted aspect list as columns
    
    Data Structures
    ---------------
    Input:
      text_list          : LIST
      aspect_dict_list   : LIST
      target_aspect_list : LIST
    Returns:
      df                 : DATA FRAME
  '''
  # data={'tweets':text_list}
  df = pd.DataFrame()
  df['tweets'] = text_list
  # df = pd.DataFrame(data)
  if target_aspect_list:
    for aspect in target_aspect_list:
      df[aspect] = np.nan

    for inner_list in aspect_dict_list:
      for _dict in inner_list:
        for key in _dict:
          if key in target_aspect_list:
            df.loc[aspect_dict_list.index(inner_list), key]=_dict[key]

  return df

def get_sentence_data_frame(text_list=None, label_list=None):
  '''
    Accepts the list of text and sentence label.
    Returns data frame with tweets as rows and sentence label 
    and tweets as columns
    
    Data Structures
    ---------------
    Input:
      text_list   : LIST
      label_list  : LIST
    Returns:
      df          : DATA FRAME
  '''
  
  if text_list:
    data={'tweets':text_list}
    df = pd.DataFrame(data)
  else:
    df = pd.DataFrame()
  
  positive = 'positive'
  negative = 'negative'
  neutral = 'neutral'
  df[positive] = 0
  df[negative] = 0
  df[neutral] = 0
  
  for i in range(len(label_list)):
    label = label_list[i]
    if label == 'pos':
      df.loc[i,positive]= 1
    elif label == 'neg':
      df.loc[i,negative]= 1
    elif label == 'neu':
      df.loc[i,neutral]= 1
  
  df = df.fillna(0)
  return df


def get_positive_data_frame(df, aspect_list):
  '''
  Returns data frame of all positive aspects in tweets and tag it as 1 
  from the data frame with all aspects
  
  Data Structures
    ---------------
    Input:
      df          : DATA FRAME
      aspect_list : LIST
    Returns:
      df          : DATA FRAME
  '''
  for aspect in aspect_list:
      df[aspect]=df[aspect].replace(['positive'], [1])
      df[aspect]=df[aspect].replace(['negative', 'neutral'],[0, 0])
  df = df.fillna(0)
  return df

def get_negative_data_frame(df, aspect_list):
  '''
  Returns data frame of all negative aspects in tweets and tag it as 1 
  from the data frame with all aspects
  
  Data Structures
    ---------------
    Input:
      df          : DATA FRAME
      aspect_list : LIST
    Returns:
      df          : DATA FRAME
  '''
  for aspect in aspect_list:
    df[aspect]=df[aspect].replace(['negative'], [1])
    df[aspect]=df[aspect].replace(['positive', 'neutral'],[0, 0])
  df = df.fillna(0)
  return df

def get_neutral_data_frame(df, aspect_list):
  '''
  Returns data frame of all neutral aspects in tweets and tag it as 1 
  from the data frame with all aspects
  
  Data Structures
    ---------------
    Input:
      df          : DATA FRAME
      aspect_list : LIST
    Returns:
      df          : DATA FRAME
  '''
  for aspect in aspect_list:
    df[aspect]=df[aspect].replace(['neutral'], [1])
    df[aspect]=df[aspect].replace(['negative', 'positive'],[0, 0])
  df = df.fillna(0)
  return df

def get_dict_aspect(y, target_aspect_list):
  '''
    Returns the dictionary containing the target aspects with the assigned 
    value if the aspect exists in y 
  
    Data Structures
    ---------------
    Input:
      y                   : NUMPY ARRAY
      target_aspect_list  : LIST
    Returns:
      df          : DATA FRAME
  '''
  
  position=[]
  for innerlist in y:
      position.append([i for i, j in enumerate(innerlist) if j == 1])
        
  dict_aspect=[]
  for innerlist in position:
    inner_dict={}
    for word in target_aspect_list:
      if target_aspect_list.index(word) in innerlist:
        inner_dict[word]=1
      else:
        inner_dict[word]=0
    dict_aspect.append(inner_dict)
  
  return dict_aspect

def get_dict_aspect_polarity(y, target_aspect_list, labelled_aspect_list, 
                            polarity=None):
  if polarity == "pos":
    polarity = "positive"
  elif polarity == "neg":
    polarity = "negative"
  elif polarity == "neu":
    polarity = "neutral"
  
  position=[]
  for innerlist in y:
    position.append([i for i, j in enumerate(innerlist) if j == 1])
  
  pos_dict_aspect=[]
  for innerlist in position:
    i = position.index(innerlist)
    inner_dict={}
    labelled_list = labelled_aspect_list[i]
    for word in target_aspect_list:
      if target_aspect_list.index(word) in innerlist:
        for prelabel_aspect in labelled_list:
          if (word in prelabel_aspect.keys() 
              and prelabel_aspect[word] == polarity):
            inner_dict[word]=5
          else:
            inner_dict[word]=0
      else:
        inner_dict[word]=0
    pos_dict_aspect.append(inner_dict)
  return pos_dict_aspect

def get_aspect_from_dict_list(aspect_dict_list):
  '''
    Returns list of list of aspects for a single tweet from all tweets
    Example: [[aspect1, aspect2], [aspect1]]
    
    Parameters
    ---
    aspect_dict_list - list of list with dict containing {aspect : label}
    
    Data Structures
    ---------------
    Input:
      aspect_dict_list  : LIST
    Returns:
      aspect_list       : LIST
  '''
  
  aspect_list = []

  try:
    for i in range(len(aspect_dict_list)):
      annotated_aspect_list = aspect_dict_list[i]
      inner_list = []
      for j in range(len(annotated_aspect_list)):
        inner_dict = annotated_aspect_list[j]
        for aspect in inner_dict:
          inner_list.append(aspect)
      aspect_list.append(inner_list)
  except Exception as e:
    print(aspect_dict_list)
    print(e)
    
  return aspect_list

def remove_duplicate_aspects(aspect_list):
  '''
    This function removes duplicate inputs in the list of inputs 
    from both annotated and extracted aspect lists
    
    Data Structures
    ---------------
    Input:
      aspect_list      : LIST
    Returns:
      removed_dup_list : LIST
  '''
  
  removed_dup_list = []
  for i in range(len(aspect_list)):
    inner_list = aspect_list[i]
    removed_dup = list(dict.fromkeys(inner_list))
    removed_dup_list.append(removed_dup)
  
  return removed_dup_list

def get_aspect_array(aspect_list, all_aspect_list):
  '''
    This function returns an array of 1s and 0s whether the aspect 
    appears in the input or not
    
    Data Structures
    ---------------
    Input:
      aspect_list     : LIST
      all_aspect_list : LIST
    Returns:
      aspect_array    : ARRAY LIST
  '''
  
  aspect_array = []
  for i in range(len(aspect_list)):
    tweet_aspect = aspect_list[i]
    tweet_aspect_array = []
    for j in range(len(all_aspect_list)):
      aspect = all_aspect_list[j]
      if aspect in tweet_aspect:
        tweet_aspect_array.append(1)
      else:
        tweet_aspect_array.append(0)
    aspect_array.append(tweet_aspect_array)
  return aspect_array

def get_user_input(user_input=None):
  '''
    Accepts user input as list then preprocess it if given. 
    If user_input is a NoneType this prompts the user to select a csv file.
    Returns the preprocessed user input list and the original user input list.

    Data Structures
    ---------------
    Input:
      - user_input  : LIST
    Output:
      - test_data               : LIST
      - preprocessed_user_input : LIST

  '''
  import pandas as pd
  test_data = user_input
  preprocessed_user_input = preprocessing.preprocess(test_data)
  return test_data,preprocessed_user_input

def get_lexicon():
  '''
    This function returns the list of negative and positive words separately
    
    Data Structures
    ---------------
    negative_opinion_words  : LIST
    positive_opinion_words  : LIST
  '''
  
  negative_opinion_words_filename = 'ensemble_analyzer/apps/public/data/resources/negative-words.txt'
  negative_opinion_words = open(os.path.join(
    cwd, negative_opinion_words_filename), 'r').read()

  positive_opinion_words_filename = 'ensemble_analyzer/apps/public/data/resources/positive-words.txt'
  positive_opinion_words = open(os.path.join(
    cwd, positive_opinion_words_filename), 'r').read()

  return (negative_opinion_words,positive_opinion_words)

def get_test_aspect_dict_list(opinion_aspect_dict_list):
  '''
    This function pre-labels the aspects depending on the opinion associated with it
    whether the opinion exist in the positive or negative words list
    
    Data Structures
    ---------------
    Input:
      opinion_aspect_dict_list  : LIST
    Returns:
      pre_labelled_aspect_list  : LIST
  '''
  
  test_aspect_dict_list = []
  negative_opinion_words,positive_opinion_words = get_lexicon()
  
  for i in range(len(opinion_aspect_dict_list)):
    inner_list = opinion_aspect_dict_list[i]
    pre_labelled_aspect = []
    for j in range(len(inner_list)):
      try:
        opinion_aspect = inner_list[j]
        for opinion,aspect in opinion_aspect.items():
          if opinion in positive_opinion_words:
            pre_labelled_aspect.append({aspect:"positive"})
          elif opinion in negative_opinion_words:
            pre_labelled_aspect.append({aspect:"negative"})
          else:
            pre_labelled_aspect.append({aspect:"neutral"})
      except:
        print(j)
        print(inner_list[j])
    test_aspect_dict_list.append(pre_labelled_aspect)
  
  return test_aspect_dict_list

def encode_label(y):
  '''
    This function encode one-hot encoded labels to categorical
    labels
    
    Data Structures
    ---------------
    Input:
      y  : LIST
    Returns:
      cat_labels  : LIST
  '''
  
  cat_labels = []

  try:
    y = y.tolist()
  except:
    y = y

  for i in range(len(y)):
    label = y[i]
    if label[0] == 1:
      cat_labels.append(1)
    elif label[1] == 1:
      cat_labels.append(-1)
    elif label[2] == 1:
      cat_labels.append(0)
    else:
      cat_labels.append(0)

  return cat_labels

