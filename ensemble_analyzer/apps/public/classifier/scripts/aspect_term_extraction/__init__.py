'''
  This module is intended for the extraction of aspect terms from user input using the rules indicated 
  in the study of Shafie et al. (2014) with additional relation 'advmod' as found that it can extract compound aspect given
  that the previous relation is nsubj and the governor of the previous relation is singular or plural noun.
  In addition, the conj relation pattern was modified to accept NN/NNS/NNP governor as it was found that it can extract
  not only singular aspect unlike in the study of Shafie et al. (2014)
'''

# Author            : Afrahly Afable
# Calling Sequence  : run_stanford_server()
#                     dep_parsing()
#                     extract_aspects()
#                     prune()
#                     pop_token()
#                     get_n_grams()
#                     get_extracted_aspects()
#                     get_ate_performance()
#                     _get_array()
#                     _binarize()
#                     aspect_extraction(user_input)
# Date Written      : November 3, 2021
# Date Revised      : November 29, 2021
# Purpose           : To extract aspects from user input for aspect term polarity classification
# Data Structures   : Input Variable/s:
#                       - user_input                : LIST
#                     Output Variable/s:
#                       - aspects                   : LIST
#                       - opinion_aspect_dict_list  : LIST
#                       - dependency                : LIST

import os
import re
import time
import subprocess
from tqdm import tqdm
from nltk.parse.corenlp import CoreNLPDependencyParser

from .ate_rules import *
from ensemble_analyzer.apps.public.classifier.scripts.utils import *
from ensemble_analyzer.apps.public.classifier.scripts.display import *
from ensemble_analyzer.apps.public.classifier.scripts.metrics import *

CWD = os.getcwd()

STOP_WORDS_FILENAME = 'ensemble_analyzer/apps/public/data/resources/stop_words.txt'
STOP_WORDS = open(os.path.join(CWD, STOP_WORDS_FILENAME), 'r').read()
STOP_WORDS = STOP_WORDS.split("\n")

class AspectTermExtraction:

  def __init__(self, preprocessed_text_list=None, mode=None):
    self.preprocessed_text_list = preprocessed_text_list
    self.mode = mode

  def run_stanford_server(self):
    '''
      This function starts the Stanford server
    
    '''
    
    command = ('java -mx1g -cp "*" ' 
              'edu.stanford.nlp.pipeline.StanfordCoreNLPServer ' 
              '-port 8000 -timeout 15000 -quiet')
    try:
      parser_path = os.path.join(CWD, 'ensemble_analyzer/apps/public/stanford-corenlp-4.3.1')
    except:
      parser_path = os.path.join(CWD, 'ensemble_analyzer/apps/public/stanford-corenlp-4.3.2')
    
    subprocess.Popen(command, cwd=parser_path, 
                    shell =True, stdout=subprocess.PIPE)
    
    print(cwd)
    print(parser_path)

    sleep_duration = 2
    time.sleep(sleep_duration)

  def dep_parsing(self, text):
    '''
      This function returns the dependencies from the
      Stanford dependency parser
      
      
      Data Structures
      ---------------
      Input:
        text        : STRING
      Returns:
        dependencies: LIST
    '''
    
    dependencies = []

    parser = CoreNLPDependencyParser(url='http://localhost:8000')

    if isinstance(text, str):
      try:
        text = text.lower()
        result = parser.raw_parse(text)
        dependency = result.__next__()
        dependencies = list(dependency.triples())
      except:
        print(text)
    elif text:
      text = ' '.join(text)
      text = text.lower()
      try:
        result = parser.raw_parse(text)
        dependency = result.__next__()
        dependencies = list(dependency.triples())
      except:
        print(text)

    return dependencies

  def extract_aspects(self, dependencies, text):
    '''
      This function acts as the main method for the aspect term
      extraction module which contains several cases that maps
      the relation from the dependency list to the extraction rule.
      
      
      Data Structures
      ---------------
      Input:
        dependencies  : LIST
        text          : STRING
      Returns:
        aspect_list               : LIST
        opinion_aspect_dict_list  : LIST
    '''
    
    relations = [
        'nsubj', 'nsubj:pass', 'amod', 'dobj', 'obj', 'obl', 'nmod',
        'nmod:poss', 'conj', 'compound', 'advmod', 'case', 'dep', 
        'aux', 'xcomp', 'compound:prt', 'csubj'
        ]
    
    temp_aspect_list = []
    opinion_list = []
    temp_opinion_aspect_dict_list = []
    
    prev_rel = ''
    prev_case = ''
    prev_gov = ''

    input_str = [text]
    five_grams = self.get_n_grams(input_str)
    
    for index, dependency in enumerate(dependencies):
      rel_aspect_list = []
      rel_opinion_list = []
      rel_opinion_aspect_dict_list = []
      
      gov = dependency[0]
      rel = dependency[1]
      dep = dependency[2]

      gov_pos = gov[1]
      gov_token = gov[0]
      dep_pos = dep[1]
      dep_token = dep[0]
      
      has_stopword = False
      has_opinion = False

      if rel in relations:
        if rel == 'nsubj':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_nsubj(gov_pos, gov_token, 
                                                        dep_pos, dep_token,
                                                        opinion_list)
        elif rel == 'amod':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_amod(gov_pos, gov_token, 
                                                      dep_pos, dep_token,
                                                      opinion_list)
          if (prev_gov == gov_token and gov_token in temp_aspect_list):
            (rel_aspect_list_mwa, rel_opinion_list_mwa,
                rel_opinion_aspect_dict_list_mwa) = rel_comp_mwa(gov_pos, gov_token, 
                                                        dep_pos, dep_token,
                                                        temp_aspect_list,opinion_list)
            rel_aspect_list.extend(rel_aspect_list_mwa)
            rel_opinion_list.extend(rel_opinion_list_mwa)
            rel_opinion_aspect_dict_list.extend(rel_opinion_aspect_dict_list_mwa)
        elif rel == 'dobj' or rel == 'obj' or rel == 'obl':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list)  = rel_dobj(gov_pos, gov_token, 
                                                      dep_pos, dep_token)
        elif rel == 'nmod':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_nmod(gov_pos, gov_token, 
                                                      dep_pos, dep_token,
                                                      opinion_list)
          if (prev_gov == gov_token and gov_token in temp_aspect_list):
            (rel_aspect_list_mwa, rel_opinion_list_mwa,
                rel_opinion_aspect_dict_list_mwa) = rel_comp_mwa(gov_pos, gov_token, 
                                                        dep_pos, dep_token,
                                                        temp_aspect_list,opinion_list)
            rel_aspect_list.extend(rel_aspect_list_mwa)
            rel_opinion_list.extend(rel_opinion_list_mwa)
            rel_opinion_aspect_dict_list.extend(rel_opinion_aspect_dict_list_mwa)
        elif rel == 'conj':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_conj(gov_pos, gov_token, 
                                                      dep_pos, dep_token,
                                                      opinion_list)
        elif rel == 'compound':
          if (prev_rel == 'case' and prev_case == 'to'):
            (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = case_to(gov_pos, gov_token, 
                                                      dep_pos, dep_token,
                                                      opinion_list)
          else:
            (rel_aspect_list, rel_opinion_list,
                rel_opinion_aspect_dict_list) = rel_comp(gov_pos, gov_token, 
                                                        dep_pos, dep_token,
                                                        opinion_list)
          if (prev_gov == gov_token and gov_token in temp_aspect_list):
            (rel_aspect_list_mwa, rel_opinion_list_mwa,
                rel_opinion_aspect_dict_list_mwa) = rel_comp_mwa(gov_pos, gov_token, 
                                                        dep_pos, dep_token,
                                                        temp_aspect_list,opinion_list)
            rel_aspect_list.extend(rel_aspect_list_mwa)
            rel_opinion_list.extend(rel_opinion_list_mwa)
            rel_opinion_aspect_dict_list.extend(rel_opinion_aspect_dict_list_mwa)
        elif rel == 'compound:prt':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list,has_opinion) = rel_comp_prt(gov_pos, gov_token, 
                                                      dep_pos, dep_token,
                                                      opinion_list)
        elif rel == 'advmod':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_advmod(gov_pos, gov_token, 
                                                      dep_pos, dep_token,
                                                      opinion_list)
        elif rel == 'case':
          prev_case = dep_token
          if (dep_token == 'of' and gov_token in temp_aspect_list):
            (rel_aspect_list, rel_opinion_list,
                rel_opinion_aspect_dict_list) = case_of(gov_pos, gov_token, 
                                                        dep_pos, dep_token, 
                                                        temp_aspect_list, 
                                                        temp_opinion_aspect_dict_list)
            if len(temp_aspect_list) > 1:
              index = temp_aspect_list.index(gov_token)
              temp_aspect_list.pop(index)
              temp_aspect_list.pop(index-1)
              temp_opinion_aspect_dict_list.pop(index)
              temp_opinion_aspect_dict_list.pop(index-1)
        elif rel == 'dep':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_dep(gov_pos, gov_token,
                                                      dep_pos, dep_token)
        elif rel == 'aux':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_aux(gov_pos, gov_token, 
                                                      dep_pos, dep_token)
        elif rel == 'xcomp':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_xcomp(gov_pos, gov_token, 
                                                        dep_pos, dep_token)
        elif rel == 'csubj':
          (rel_aspect_list, rel_opinion_list,
              rel_opinion_aspect_dict_list) = rel_csubj(gov_pos, gov_token, 
                                                        dep_pos, dep_token)

        prev_gov = gov_token

        for a in rel_aspect_list:
          if len(a.split()) > 1:
            tokens = a.split()
            for token in tokens:
              if token in STOP_WORDS:
                has_stopword = True
                break
              else:
                continue
          else:
            if a in STOP_WORDS:
              has_stopword = True

        if not has_stopword or has_opinion:
          opinion_list.extend(rel_opinion_list)
          temp_aspect_list.extend(rel_aspect_list)
          temp_opinion_aspect_dict_list.extend(rel_opinion_aspect_dict_list)

      prev_rel = rel

    temp_aspect_list = list(dict.fromkeys(temp_aspect_list))
    (temp_aspect_list, 
        temp_opinion_aspect_dict_list) = self.prune(temp_aspect_list, 
                                                    temp_opinion_aspect_dict_list)

    aspect_list = []
    opinion_aspect_dict_list = []

    for term in temp_aspect_list:
      if term not in five_grams:
        continue
      else:
        index = temp_aspect_list.index(term)
        aspect_list.append(term)
        opinion_aspect_dict_list.append(temp_opinion_aspect_dict_list[index])

    return (aspect_list, opinion_aspect_dict_list)

  def prune(self, aspect_list, opinion_aspect_dict_list):
    '''
      This function removes aspect terms that
      was extracted but is either nonsense or irrelevant
      and its corresponding opinion and aspect dictionary
      
      Data Structures
      ---------------
      Input:
        aspect_list               : LIST
        opinion_aspect_dict_list  : LIST
      Returns:
        aspect_list               : LIST
        opinion_aspect_dict_list  : LIST
    '''
    
    temp_aspect_list = aspect_list
    
    for aspect in temp_aspect_list:
      if len(aspect.split()) > 1:
        terms = aspect.split()
        for term in terms:
          if term in aspect_list:
            term_index = aspect_list.index(term)
            aspect_list.pop(term_index)
            opinion_aspect_dict_list.pop(term_index)

    return (aspect_list, opinion_aspect_dict_list)

  def pop_token(self, aspect_list, opinion_aspect_dict_list, dep_token, 
                gov_token):
    '''
      This function pops token from the argument lists given the
      condition
      
      Data Structures
      ---------------
      Input:
        aspect_list               : LIST
        opinion_aspect_dict_list  : LIST
        dep_token                 : STRING
        gov_token                 : STRING
      Returns:
        aspect_list               : LIST
        opinion_aspect_dict_list  : LIST
    '''
    
    if dep_token in aspect_list:
      aspect_index = aspect_list.index(dep_token)
      aspect_list.pop(aspect_index)
      opinion_aspect_dict_list.pop(aspect_index)
    if gov_token in aspect_list:
      aspect_index = aspect_list.index(gov_token)
      aspect_list.pop(aspect_index)
      opinion_aspect_dict_list.pop(aspect_index)

    return (aspect_list, opinion_aspect_dict_list)

  def get_n_grams(self, tokens):
    '''
      This function gets the text n-grams where n = 5, to check whether
      the extracted multi-word aspects do exist in the given text grams
      
      Data Structures
      ---------------
      Input:
        tokens               : LIST
      Returns:
        grams               : LIST
    '''
    
    tokens = re.sub(r'[^a-zA-Z0-9-\s]', ' ', ' '.join(tokens).lower())
    tokens = tokens.split()
    grams = []

    for i in range(0,5):
      temp=zip(*[tokens[i:] for i in range(i)])
      grams.extend(' '.join(ngram) for ngram in temp)

    return grams

  def get_extracted_aspects(self):
    '''
      This function returns the extracted aspects from the preprocessed user input
      as well as the list of dictionary containing opinion(key) and aspect(value) pairs
      
      Data Structures
      ---------------
      Input:
        self.preprocessed_text_list : LIST
      Returns:
        extracted_aspect_list       : LIST
        opinion_aspect_dict_list    : LIST
        dependency_list             : LIST
    '''

    extracted_aspect_list = []
    opinion_aspect_dict_list = []
    dependency_list = []

    # self.run_stanford_server()
    print("\nEXTRACTING ASPECTS ...")

    for text in tqdm(self.preprocessed_text_list):
      dependencies = self.dep_parsing(text)

      (extracted_aspects,
      opinion_aspect_dict) = self.extract_aspects(dependencies, text)

      dependency_list.append(dependencies)
      extracted_aspect_list.append(extracted_aspects)
      opinion_aspect_dict_list.append(opinion_aspect_dict)

    if self.mode == "test":
      return (extracted_aspect_list, opinion_aspect_dict_list, 
              dependency_list)
    elif self.mode == "implement":
      test_aspect_dict_list = get_test_aspect_dict_list(opinion_aspect_dict_list)
      return (extracted_aspect_list, opinion_aspect_dict_list, 
              dependency_list, test_aspect_dict_list)

  def get_ate_performance(self, annotated_list, extracted_list):
    '''
      This function returns performance of the proposed
      aspect term extraction method
      
      Data Structures
      ---------------
      Input:
        annotated_list : LIST
        extracted_list : LIST
      Returns:
        accuracy  : FLOAT
        precision : FLOAT
        recall    : FLOAT
        f1        : FLOAT
    '''
    
    (annotated_array_list, 
        extracted_array_list) = self._get_array(annotated_list,
                                                extracted_list)
    (accuracy, precision, 
        recall, f1) = get_al_metrics(annotated_array_list,
                                    extracted_array_list,
                                    "micro")

    print("==============================================================")
    print("        ASPECT TERM EXTRACTION PERFORMANCE EVALUATION         ")
    print("==============================================================")
    display_metrics(accuracy, precision, recall, f1)

    return (accuracy, precision, recall, f1)
  
  def _get_array(self, annotated_list, extracted_list):
    '''
      This function returns an array of 1s and 0s used in
      the performance computation of the aspect term extraction
      method
      
      Data Structures
      ---------------
      Input:
        annotated_list : LIST
        extracted_list : LIST
      Returns:
        annotated_array_list  : LIST
        extracted_array_list  : LIST
    
    '''
    
    annotated_array_list = []
    extracted_array_list = []
    
    for i in range(len(annotated_list)):
      annotated = sorted(annotated_list[i])
      extracted = sorted(extracted_list[i])

      combined = [*annotated, *extracted]
      combined = list(dict.fromkeys(combined))
      combined = sorted(combined)

      if annotated:
        annotated_array = self._binarize(annotated, annotated)
        extracted_array = self._binarize(extracted, annotated)
      else:
        annotated_array = [0]
        extracted_array = [0]

      annotated_array_list.append(annotated_array)
      extracted_array_list.append(extracted_array)

    return (annotated_array_list, extracted_array_list)

  def _binarize(self, aspect_list, all_aspect_list):
    '''
      This function binarizes the given argument lists used
      in getting the array of the given list
      
      Data Structures
      ---------------
      Input:
        aspect_list     : LIST
        all_aspect_list : LIST
      Returns:
        array  : LIST    
    '''
    
    array = []
    temp_aspect_list = []
    temp_all_aspect_list = []

    for term in all_aspect_list:
      if len(term.split()) > 1:
        for t in term.split():
          temp_all_aspect_list.append(t)
      else:
        temp_all_aspect_list.append(term)

    for term in aspect_list:
      if len(term.split()) > 1:
        for t in term.split():
          temp_aspect_list.append(t)
      else:
        temp_aspect_list.append(term) 

    temp_all_aspect_list = sorted(temp_all_aspect_list)
    temp_aspect_list = sorted(temp_aspect_list)

    for aspect in temp_all_aspect_list:
      if aspect in temp_aspect_list:
        array.append(1)
      else:
        array.append(0)

    return array