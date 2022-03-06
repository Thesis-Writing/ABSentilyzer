# Program Title         : ate_rules.py
# Author                : Afrahly Afable
# General System Design : This module is intended for the extraction of 
#                         aspect terms from user input using the rules 
#                         indicated in the study of Shafie et al. (2014)
#                         with additional modifications in relations and rules.
#                         for implementation.
# Date Written          : November 3, 2021
# Date Revised          : November 29, 2021
# Purpose               : To extract aspects from user input for
#                         aspect term polarity classification
# Data Structures       : List, Dictionary, String


import os

CWD = os.getcwd()
OPINION_WORDS_FILENAME = 'ensemble_analyzer/apps/public/'\
                          'data/resources/opinion_words.txt'
OPINION_WORDS = open(os.path.join(CWD, OPINION_WORDS_FILENAME), 'r').read()
OPINION_WORDS = OPINION_WORDS.split("\n")

__all__ = [
        "rel_nsubj", "rel_amod", "rel_dobj", "rel_nmod", "rel_conj",
        "rel_comp", "rel_advmod", "rel_case", "case_of", "case_to",
        "rel_dep", "rel_aux", "rel_xcomp", "rel_comp_mwa", "rel_comp_prt",
        "rel_csubj"
        ]

def rel_nsubj(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []
  
  if (gov_pos.startswith('J') 
      and dep_pos.startswith('N')) or (gov_pos.startswith('N') 
                                      and dep_pos.startswith('N')):
    if gov_token not in OPINION_WORDS:
      if opinion_list:
        opinion = opinion_list[len(opinion_list)-1]
        rel_opinion_list.append(opinion)
      rel_aspect_list.append(gov_token  + " " + dep_token)
      rel_opinion_aspect_dict_list.append({opinion:gov_token  + " " + dep_token})
    else:
      rel_opinion_list.append(gov_token)
      rel_aspect_list.append(dep_token)
      rel_opinion_aspect_dict_list.append({gov_token:dep_token})
  if gov_pos.startswith('V') and dep_pos.startswith('N'):
    rel_opinion_list.append(gov_token)
    rel_aspect_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({gov_token:dep_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_amod(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []
  
  if gov_pos.startswith('N') and dep_pos.startswith('J'):
    if opinion_list:
      opinion = opinion_list[len(opinion_list)-1]
      rel_opinion_list.append(opinion)
    if dep_token not in OPINION_WORDS:
      rel_aspect_list.append(dep_token  + " " + gov_token)
      rel_opinion_aspect_dict_list.append({opinion:dep_token  + " " + gov_token})
    else:
      rel_opinion_list.append(dep_token)
      rel_aspect_list.append(gov_token)
      rel_opinion_aspect_dict_list.append({dep_token:gov_token})
  elif gov_pos.startswith('N') and dep_pos.startswith('V'):
    if opinion_list:
      opinion = opinion_list[len(opinion_list)-1]
      rel_opinion_list.append(opinion)
    if dep_token not in OPINION_WORDS:
      rel_aspect_list.append(dep_token  + " " + gov_token)
      rel_opinion_aspect_dict_list.append({opinion:dep_token  + " " + gov_token})
    else:
      rel_aspect_list.append(dep_token)
      rel_opinion_list.append(gov_token)
      rel_opinion_aspect_dict_list.append({dep_token:gov_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list) 

def rel_dobj(gov_pos,gov_token,dep_pos,dep_token):
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  if gov_pos.startswith('V') and dep_pos.startswith('N'):
    rel_opinion_list.append(gov_token)
    rel_aspect_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({gov_token:dep_token})
  elif (gov_pos.startswith('J') 
        and dep_pos.startswith('N')) or (gov_pos.startswith('N') 
                                        and dep_pos.startswith('N')):
    if gov_token in OPINION_WORDS:
      rel_opinion_list.append(gov_token)
      rel_aspect_list.append(dep_token)
      rel_opinion_aspect_dict_list.append({gov_token:dep_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_nmod(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  if opinion_list:
    opinion = opinion_list[len(opinion_list)-1]
    rel_opinion_list.append(opinion)
  if (gov_pos == 'NN') and (dep_pos in ['NN', 'NNS']):
    rel_aspect_list.extend((gov_token,dep_token))
    rel_opinion_aspect_dict_list.append({opinion:gov_token})
    rel_opinion_aspect_dict_list.append({opinion:dep_token})
  elif gov_pos == 'JJ' and (dep_pos in ['NN', 'NNS']):
    rel_opinion_list.append(gov_token)
    rel_aspect_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({gov_token:dep_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_conj(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []
  
  if gov_pos.startswith('N') and dep_pos.startswith('N'):
    if opinion_list:
      opinion = opinion_list[len(opinion_list)-1]
      rel_opinion_list.append(opinion)
    rel_aspect_list.extend((gov_token,dep_token))
    rel_opinion_aspect_dict_list.append({opinion:gov_token})
    rel_opinion_aspect_dict_list.append({opinion:dep_token})
  elif gov_pos.startswith('N') and (dep_pos == 'JJ'):
    rel_opinion_list.append(dep_token)
    rel_aspect_list.append(gov_token)
    rel_opinion_aspect_dict_list.append({dep_token:gov_token})
  elif gov_pos.startswith('N') and (dep_pos == 'VBZ'):
    rel_opinion_list.append(dep_token)
    rel_aspect_list.append(gov_token)
    rel_opinion_aspect_dict_list.append({dep_token:gov_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_comp(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  if gov_pos.startswith('NN') and dep_pos in ['NN','NNS']:
    if dep_token in OPINION_WORDS:
      rel_aspect_list.append(gov_token)
      rel_opinion_list.append(dep_token)
      rel_opinion_aspect_dict_list.append({dep_token:gov_token})
    else:
      if opinion_list:
        opinion = opinion_list[len(opinion_list)-1]
        rel_opinion_list.append(opinion)
      rel_aspect_list.append(dep_token  + " " + gov_token)
      rel_opinion_aspect_dict_list.append({opinion:dep_token  + " " + gov_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_comp_prt(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  has_opinion = False
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  if opinion_list:
    opinion = opinion_list[len(opinion_list)-1]
    rel_opinion_list.append(opinion)
    has_opinion = True if opinion in OPINION_WORDS else False
  rel_aspect_list.append(gov_token  + " " + dep_token)
  rel_opinion_aspect_dict_list.append({opinion:gov_token  + " " + dep_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list,
          has_opinion)

def rel_comp_mwa(gov_pos,gov_token,dep_pos,dep_token,aspect_list,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  term = aspect_list[len(aspect_list)-1]
  if gov_token in term.split():
    term = term.split()
    index = term.index(gov_token) # insert dep_token before gov_token
    term.insert(index,dep_token)
    if opinion_list:
      opinion = opinion_list[len(opinion_list)-1]
      rel_opinion_list.append(opinion)
    rel_aspect_list.append(' '.join(term))
    rel_opinion_aspect_dict_list.append({opinion:' '.join(term)})

    for term in rel_aspect_list:
      if len(term.split()) > 1:
        for t in term.split():
          if t in OPINION_WORDS:
            rel_aspect_list = []
            rel_opinion_list = []
            rel_opinion_aspect_dict_list = []

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_advmod(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  if gov_pos.startswith('N') and dep_pos == 'RB':
    if dep_token not in OPINION_WORDS:
      if opinion_list:
        opinion = opinion_list[len(opinion_list)-1]
        rel_opinion_list.append(opinion)
      rel_aspect_list.append(dep_token + " " + gov_token)
      rel_opinion_aspect_dict_list.append({opinion:dep_token + " " + gov_token})
  elif dep_pos.startswith('J'):
    rel_aspect_list.append(gov_token)
    rel_opinion_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({dep_token:gov_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_case(gov_pos,gov_token,dep_pos,dep_token):
  aspect_list = []
  opinion_aspect_dict_list = []
  if gov_pos.startswith('N'):
    aspect = gov_token
    opinion = ''
    aspect_list.append(aspect)
    opinion_aspect_dict_list.append({opinion:aspect})

  return (aspect_list,
          opinion_aspect_dict_list)

def case_of(gov_pos, 
            gov_token,
            dep_pos, 
            dep_token, 
            aspect_list, 
            opinion_aspect_dict_list) :
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  index = aspect_list.index(gov_token)
  first_word = aspect_list[index-1]
  sec_word = aspect_list[index]
  aspect = first_word + " " + dep_token + " " + sec_word

  for inner_dict in opinion_aspect_dict_list:
    key = list(inner_dict.keys())[0]
    if inner_dict[key] == gov_token:
      opinion = key
      rel_opinion_list.append(opinion)
      break

  rel_aspect_list.append(aspect)
  rel_opinion_aspect_dict_list.append({opinion:aspect})
    
  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def case_to(gov_pos,gov_token,dep_pos,dep_token,opinion_list):
  opinion = ''
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []
  
  if gov_pos.startswith('N') and dep_pos.startswith('N'):
    if opinion_list:
      opinion = opinion_list[len(opinion_list)-1]
      rel_opinion_list.append(opinion)
    rel_aspect_list.append(dep_token + " " + "to" + " " + dep_token + 
                          " " + gov_token)
    rel_opinion_aspect_dict_list.append({opinion:dep_token + " " + "to" + 
                                        " " + dep_token + " " + gov_token})
  
  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_dep(gov_pos,gov_token,dep_pos,dep_token):
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []
  
  if (gov_pos.startswith('J') or gov_pos.startswith('V') 
      and dep_pos.startswith('N')):
    rel_opinion_list.append(gov_token)
    rel_aspect_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({gov_token:dep_token})
  
  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)


def rel_aux(gov_pos,gov_token,dep_pos,dep_token):
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []
  
  if gov_pos == 'JJ':
    rel_opinion_list.append(gov_token)
    rel_aspect_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({gov_token:dep_token})
  
  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)


def rel_xcomp(gov_pos,gov_token,dep_pos,dep_token):
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []
  
  if gov_pos.startswith('J'):
    rel_opinion_list.append(gov_token)
    rel_aspect_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({gov_token:dep_token})
  
  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)

def rel_csubj(gov_pos,gov_token,dep_pos,dep_token):
  rel_aspect_list = []
  rel_opinion_list = []
  rel_opinion_aspect_dict_list = []

  if gov_pos.startswith('J') and dep_pos.startswith('N'):
    rel_opinion_list.append(gov_token)
    rel_aspect_list.append(dep_token)
    rel_opinion_aspect_dict_list.append({gov_token:dep_token})

  return (rel_aspect_list, 
          rel_opinion_list, 
          rel_opinion_aspect_dict_list)


def remove_aspect(aspect_list,opinion_aspect_dict_list,aspect):
  temp_aspect_list = []
  temp_opinion_dict_list = []
  position = None
  for i in range(len(aspect_list)):
    temp_aspect = aspect_list[i]
    opinion_aspect_dict = opinion_aspect_dict_list[i]
    if not temp_aspect == aspect:
      temp_aspect_list.append(temp_aspect)
      temp_opinion_dict_list.append(opinion_aspect_dict)
    else:
      position = i
      if temp_aspect_list and temp_opinion_dict_list:
        temp_aspect_list.pop()
        temp_opinion_dict_list.pop()
      
  return (temp_aspect_list,
          temp_opinion_dict_list,
          position)