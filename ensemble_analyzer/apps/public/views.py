# -*- coding: utf-8 -*-

'''
    This module interacts with the backend by taking a request and by returning a response.
'''

# Author            : Keith Barrientos
#                     Afrahly Afable
#                     John Edrick Allas
# Calling Sequence  : index(request: HttpRequest) -> HttpResponse
#                     home(request: HttpRequest) -> HttpResponse
#                     how(request: HttpRequest) -> HttpResponse
#                     about(request: HttpRequest) -> HttpResponse
# Date Written      : October 5, 2021
# Date Revised      : December 16, 2021
# Purpose           : Respond to HTTP Request methods 



from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from .forms import InputForm
from ensemble_analyzer.apps.public.absentilyzer import ABSentilyzer

from pandas import *
import os


def index(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        context = {}
        form = InputForm(request.POST, request.FILES)

        if form.is_valid():
            outText = form.cleaned_data.get("text")
            outCSV = form.cleaned_data.get("csv")
            
            if outCSV:
                print(outCSV.name) #prints the file name in the console

                # file type validation
                ext = os.path.splitext(outCSV.name)[1] # returns path+filename
                valid_filetype = ['.csv']
                if ext.lower() in valid_filetype:
                    #read uploaded csv
                    data = read_csv(outCSV)

                    # the system has a note that the column the user wants to classify must be renamed as "input"
                    # convert "input column into list"
                    included_col = data['input'].tolist()

                    # check no. of rows
                    check_rows = (len(included_col))

                    # number of rows validation
                    if check_rows <= 499:
                        # proceed to backend
                        main_table_dict = {}
                        preprocessed_table_dict = {}
                        final_sentence_polarity_table_dict = {}
                        context['form'] = form

                        # outCSV must be a list
                        analyzer = ABSentilyzer(included_col)
                        analyzer.process_input()
                        
                        (main_table_dict,
                            preprocessed_table_dict,
                            final_sentence_polarity_table_dict) = analyzer.get_tables()

                        return render(request, "index.html", 
                                    {'form': form, 
                                        'main_table_dict': main_table_dict, 
                                        'preprocessed_table_dict': preprocessed_table_dict, 
                                        'final_sentence_polarity_table_dict': final_sentence_polarity_table_dict,
                                        'csv_allowed':'yes'})

                    # else if row is greater than 500, raise csv_rowError validation then reload page
                    else:
                        form = InputForm()
                        return render(request, "index.html", {'form': form, 'csv_rowError': True})
                
                # else if filetype is not supported, raise filetype_error validation then reload page
                else:
                    form = InputForm()
                    return render(request, "index.html", {'form': form, 'filetype_error': True})            

            elif outText:
                main_table_dict = {}
                preprocessed_table_dict = {}
                final_sentence_polarity_table_dict = {}
                user_input = []
                user_input.append(outText)
                
                context['form'] = form
                context['main_table_dict'] = main_table_dict
                context['preprocessed_table_dict'] = preprocessed_table_dict
                context['final_sentence_polarity_table_dict'] = final_sentence_polarity_table_dict

                analyzer = ABSentilyzer(user_input)
                analyzer.process_input()

                (main_table_dict,
                    preprocessed_table_dict,
                    final_sentence_polarity_table_dict) = analyzer.get_tables()
                
                return render(request, "index.html", 
                            {'form': form, 
                            'main_table_dict': main_table_dict,
                            'preprocessed_table_dict': preprocessed_table_dict,
                            'final_sentence_polarity_table_dict': final_sentence_polarity_table_dict,
                            'text_allowed':'yes'})

        # else if there is no input, raise no_input validation then reload page
        else:
            form = InputForm()
            return render(request, "index.html", {'form': form, 'no_input': True})

    else:
        form = InputForm()
        return render(request, "index.html", {'form': form})


def home(request: HttpRequest) -> HttpResponse:
    return render(request, "index.html")


def how(request: HttpRequest) -> HttpResponse:
    return render(request, "how.html")


def about(request: HttpRequest) -> HttpResponse:
    return render(request, "about.html")