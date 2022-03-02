# Author            : Afrahly Afable
# Calling Sequence  : check_NoneType_tweet(text)
#                     check_no_aspect_input(original_input_list,preprocessed_input_list,extracted_aspect_list,opinion_aspect_dict_list,dependency_list,pre_labelled_aspects)
#                     check_tweet(test_list,test_aspect_dict_list,test_aspect_list)
# Date Written      : November 10, 2021
# Date Revised      : December 27, 2021
# Purpose           : To maintain validation functions used in different modules

def check_NoneType_tweet(text):
  '''
    Checks if the tweet is a None type
    
    Data Structures
    ---------------
    Input:
      text   : STRING
    Returns:
      text   : STRING
  '''
  if not isinstance(text,str):
    text = ''
  return text

def check_no_aspect_input(original_input_list, preprocessed_input_list, 
                          extracted_aspect_list, opinion_aspect_dict_list, 
                          dependency_list, pre_labelled_aspects):
  '''
    Remove inputs where there are no extracted aspects given by the ATE rules
    
    Data Structures
    ---------------
    Input:
      original_input_list       : LIST
      preprocessed_input_list   : LIST
      extracted_aspect_list     : LIST
      opinion_aspect_dict_list  : LIST  
      dependency_list           : LIST
      pre_labelled_aspects      : LIST
    Returns:
      temp_original_input_list      : LIST  
      temp_preprocessed_input_list  : LIST  
      temp_extracted_aspect_list    : LIST  
      temp_opinion_aspect_dict_list : LIST  
      temp_dependency_list          : LIST
      temp_pre_labelled_aspects     : LIST  
  '''
  
  empty_opinion_aspect_dict_list = []
  
  for i in range(len(opinion_aspect_dict_list)):
    opinion_aspect = opinion_aspect_dict_list[i]
    if not opinion_aspect:
      empty_opinion_aspect_dict_list.append(i)

  temp_original_input_list = []
  temp_preprocessed_input_list = []
  temp_extracted_aspect_list = []
  temp_opinion_aspect_dict_list = []
  temp_dependency_list = []
  temp_pre_labelled_aspects = []
  
  for i in range(len(preprocessed_input_list)):
    if i not in empty_opinion_aspect_dict_list:
      temp_original_input_list.append(original_input_list[i])
      temp_preprocessed_input_list.append(preprocessed_input_list[i])
      temp_extracted_aspect_list.append(extracted_aspect_list[i])
      temp_opinion_aspect_dict_list.append(opinion_aspect_dict_list[i])
      temp_dependency_list.append(dependency_list[i])
      temp_pre_labelled_aspects.append(pre_labelled_aspects[i])

  return (temp_original_input_list, temp_preprocessed_input_list, 
          temp_extracted_aspect_list, temp_opinion_aspect_dict_list, 
          temp_dependency_list, temp_pre_labelled_aspects)

def check_tweet(original_test_text_list, test_list,
                dependency_list, opinion_aspect_dict_list, 
                extracted_aspect_list):
  '''
    Remove inputs where there are no extracted aspects given by the ATE rules
    
    Data Structures
    ---------------
    Input:
      original_test_text_list
      test_list                    : LIST
      dependency_list              : LIST
      opinion_aspect_dict_list     : LIST
      test_aspect_dict_list        : LIST  
      extracted_aspect_list        : LIST  
    Returns:
      temp_preprocessed_test_list  : LIST
      temp_dependency_list         : LIST
      temp_opinion_aspect_dict_list: LIST
      test_aspect_dict_list        : LIST  
      temp_test_aspect_list        : LIST  
  '''
  
  empty_aspect_list = []
  
  for i in range(len(extracted_aspect_list)):
    aspects = extracted_aspect_list[i]
    if not aspects:
      empty_aspect_list.append(i)

  temp_original_test_text_list =[]
  temp_preprocessed_test_list = []
  temp_dependency_list = []
  temp_opinion_aspect_dict_list = []
  temp_test_aspect_list = []
  
  for i in range(len(test_list)):
    if i not in empty_aspect_list:
      temp_original_test_text_list.append(original_test_text_list[i])
      temp_preprocessed_test_list.append(test_list[i])
      temp_dependency_list.append(dependency_list[i])
      temp_opinion_aspect_dict_list.append(opinion_aspect_dict_list[i])
      temp_test_aspect_list.append(extracted_aspect_list[i])


  return (temp_original_test_text_list, temp_preprocessed_test_list,
          temp_dependency_list, temp_opinion_aspect_dict_list,
          temp_test_aspect_list)

def check_instance(var):
  '''
		This function checks if the instance of passed variable
    is list

		Data Structures
		---------------
		Input:
			var : LIST
		Output:
			instance  : LIST
  '''
  
  if isinstance(var[0], list):
    instance = var[0]
  return instance

def is_equal_zero(true,pred):
  '''
		This function checks if both true and predicted values
    only contain 0 values

		Data Structures
		---------------
		Input:
			true : LIST
      pred : LIST
		Output:
			True/False  : BOOLEAN
  '''

  if true.count(true[0]) == len(true):
    if pred.count(pred[0]) == len(pred):
      if true[0] == pred[0] == 0:
        return True
  else:
    return False