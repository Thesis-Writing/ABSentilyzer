'''
  This module is intended for the computation of the final aspect polarity of the sentence
  from the polarities of the aspect terms
'''

# Author            : Afrahly Afable
# Calling Sequence  : get_final_polarities(test_aspect_list,pos_ensemble_prob_list,neg_ensemble_prob_list,neu_ensemble_prob_list)
#                     get_aspect_final_polarity(pos_prob,neg_prob,neu_prob)
#                     get_sentence_polarity(final_aspect_polarity_list)
# Date Written      : December 1, 2021
# Date Revised      : December 27, 2021
# Purpose           : Compute for the final aspect term polarity

def get_aspect_polarity(test_aspect_list, pos_ensemble_prob_list, 
                        neg_ensemble_prob_list, neu_ensemble_prob_list):
  '''
      This function gets the final polarity of aspects from input
      
      Data Structures
      ---------------
      Input:
        - test_aspect_list        : LIST
        - pos_ensemble_prob_list  : LIST
        - neg_ensemble_prob_list  : LIST
        - neu_ensemble_prob_list  : LIST
      Returns:
        - final_aspect_polarity_list  : LIST
  '''
  
  final_aspect_polarity_list = []
  
  for i in range(len(test_aspect_list)):
    pos_prob_list = pos_ensemble_prob_list[i]
    neg_prob_list = neg_ensemble_prob_list[i]
    neu_prob_list = neu_ensemble_prob_list[i]

    final_input_aspect_polarity = []

    if len(test_aspect_list[i]) == 1:
      pos_prob = pos_prob_list[0][1]
      neg_prob = neg_prob_list[0][1]
      neu_prob = neu_prob_list[0][1]

      final_aspect_polarity = get_aspect_final_polarity(pos_prob, neg_prob, 
                                                        neu_prob)
      final_input_aspect_polarity.append(final_aspect_polarity)
    else:
      for j in range(len(pos_prob_list)):
        pos_prob = pos_prob_list[j]
        neg_prob = neg_prob_list[j]
        neu_prob = neu_prob_list[j]
        final_aspect_polarity = get_aspect_final_polarity(pos_prob, neg_prob, 
                                                          neu_prob)
        final_input_aspect_polarity.append(final_aspect_polarity)

    final_aspect_polarity_list.append(final_input_aspect_polarity)

  return final_aspect_polarity_list

def get_aspect_final_polarity(pos_prob, neg_prob, neu_prob):
  '''
    This function gets the final polarity of aspect by comparing 
    its pos, neg, and neu probability
    
    Data Structures
      ---------------
      Input:
        - pos_prob : FLOAT 
        - neg_prob : FLOAT
        - neu_prob : FLOAT
      Returns:
        - final_polarity  : STRING
  '''
  
  final_polarity = 'neutral'
  
  if pos_prob > neg_prob and pos_prob > neu_prob:
    final_polarity = 'positive'
  elif neg_prob > pos_prob and neg_prob > neu_prob:
    final_polarity = 'negative'
  elif neu_prob > pos_prob and neu_prob > neg_prob:
    final_polarity = 'neutral'
  elif pos_prob == neu_prob and pos_prob != 0.0:
    final_polarity = 'positive'
  elif neg_prob == neu_prob and neg_prob != 0.0:
    final_polarity = 'negative'
  elif pos_prob == neg_prob and pos_prob != 0.0:
    final_polarity = 'neutral'
  elif pos_prob == neg_prob and neg_prob == neu_prob:
    final_polarity = 'neutral'
  
  return final_polarity

def get_sentence_senti_scores(pos_ensemble_prob_list, neg_ensemble_prob_list,
                              neu_ensemble_prob_list):
  senti_score_list = []
  
  for i in range(len(pos_ensemble_prob_list)):
    inner_list = []
    
    pos_prob_list = pos_ensemble_prob_list[i]
    neg_prob_list = neg_ensemble_prob_list[i]
    neu_prob_list = neu_ensemble_prob_list[i]

    pos_score = get_score(pos_prob_list)
    neg_score = get_score(neg_prob_list)
    neu_score = get_score(neu_prob_list)

    inner_list.extend((pos_score,neg_score,neu_score))
    senti_score_list.append(inner_list)

  return senti_score_list

def get_score(prob_list):
  score = 0.0
  temp_score = 0.0

  try:
    if isinstance(prob_list[0],list):
      score = prob_list[0][1]
    else:
      for i in range(len(prob_list)):
        prob = prob_list[i]
        temp_score += prob
      score = temp_score/len(prob_list)
  except Exception as e:
    print(e)
  
  return score

def get_sentence_polarity(senti_score_list):
  '''
    This function gets the final aspect polarity list then returns the
    final polarity of the sentence
    
    Data Structures
      ---------------
      Input:
        - final_aspect_polarity_list : LIST
      Returns:
        - final_polarity_list  : LIST
  '''
  
  final_polarity_list = []
  sentence_polarity = 'neu'
  
  for i in range(len(senti_score_list)):
    pos_score = senti_score_list[i][0]
    neg_score = senti_score_list[i][1]
    neu_score = senti_score_list[i][2]

    if pos_score > neg_score and pos_score > neu_score:
      sentence_polarity = 'pos'
    elif neg_score > pos_score and neg_score > neu_score:
      sentence_polarity = 'neg'
    elif neu_score > pos_score and neu_score > neg_score:
      sentence_polarity = 'neu'

    final_polarity_list.append(sentence_polarity)

  return final_polarity_list