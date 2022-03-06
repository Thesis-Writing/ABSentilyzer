# Author            : Keith Barrientos
#                     Afrahly Afable
# Date Written      : October 5, 2021
# Date Revised      : December 16, 2021
# Purpose           : This module interacts with the backend by 
#                     taking a request and by returning a response.
#                     Respond to HTTP Request methods
#  



from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from .forms import InputForm
from ensemble_analyzer.apps.public.absentilyzer import ABSentilyzer

from pandas import *
import os

# calls and display the index.html template as well as its outputs
def index(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        context = {}
        csv_rowError = False
        not_input = False
        form = InputForm(request.POST, request.FILES)
        
        # if there is an input
        if form.is_valid():
            
            isText = form.check_text()
            isCSV = form.check_csv()
            
            # if input is CSV
            if isCSV:
                
                # calls the function that check no. of rows 
                # and if the column header is "input"
                included_col, check_rows = form.get_csv()

                main_table_dict = {}
                final_sentence_polarity_table_dict = {}
                context['form'] = form

                # outCSV must be a list
                analyzer = ABSentilyzer(included_col)
                analyzer.process_input()
                
                (main_table_dict,
                    final_sentence_polarity_table_dict) = analyzer.get_tables()
                
                aspect_dict = get_most_common_aspect(main_table_dict)
                has_most_common = False
                if len(set(aspect_dict.values()))!=1:
                    has_most_common = "True"
                else:
                    has_most_common = "False"
                
                sentiment_count_dict = get_sentiment_count(final_sentence_polarity_table_dict)
                
                return render(request, "index.html", 
                            {'form': form, 
                                'main_table_dict': main_table_dict, 
                                'aspect_dict': aspect_dict,
                                'has_most_common': has_most_common,
                                'sentiment_count_dict': sentiment_count_dict,
                                'csv_allowed':'yes',
                                'row_length':check_rows})
            
            # elif input is Text
            elif isText:
                
                inputText = form.get_text()
                
                main_table_dict = {}
                final_sentence_polarity_table_dict = {}
                user_input = []
                user_input.append(inputText)
                
                context['form'] = form
                context['main_table_dict'] = main_table_dict

                analyzer = ABSentilyzer(user_input)
                analyzer.process_input()
                (main_table_dict, 
                    final_sentence_polarity_table_dict) = analyzer.get_tables()
                
                return render(request, "index.html", 
                            {'form': form, 
                            'main_table_dict': main_table_dict,
                            'text_allowed':'yes'})
                
        elif form.isNotLarge:
            form = InputForm()
            csv_rowError = True
            return render(request, "index.html", {'form': form, 'csv_rowError':csv_rowError})
        elif form.isNotInput:
            form = InputForm()
            not_input = True
            return render(request, "index.html", {'form': form, 'not_input':not_input})
        else:
            form = InputForm()
            no_input = True
            return render(request, "index.html", {'form': form, 'no_input':no_input})
    else:
        form = InputForm()
        return render(request, "index.html", {'form': form})

# calls and displays index.html
def home(request: HttpRequest) -> HttpResponse:
    return render(request, "index.html")

# calls and displays features.html
def features(request: HttpRequest) -> HttpResponse:
    return render(request, "features.html")

# calls and displays about.html
def about(request: HttpRequest) -> HttpResponse:
    return render(request, "about.html")

# calls and displays the function that gets the most common aspects
def get_most_common_aspect(main_table_dict):
    import collections
    import operator

    aspect_list = []
    for key, values in main_table_dict.items():
        for aspect in values[1]:
            aspect_list.append(aspect)
    occurrences = collections.Counter(aspect_list)
    aspect_dict = dict(occurrences)
    aspect_dict = dict( sorted(aspect_dict.items(), key=operator.itemgetter(1),reverse=True))
    return aspect_dict

# calls and displays the function that gets the sentiment count
def get_sentiment_count(final_sentence_polarity_table_dict):
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    sentiment_count_dict = {}
    
    for key, value in final_sentence_polarity_table_dict.items():
        polarity = value[1]
        if polarity == 'pos':
            pos_count += 1
        elif polarity == 'neg':
            neg_count += 1
        elif polarity == 'neu':
            neu_count += 1
    
    sentiment_count_dict["Positive"] = pos_count
    sentiment_count_dict["Negative"] = neg_count
    sentiment_count_dict["Neutral"] = neu_count
    
    return sentiment_count_dict