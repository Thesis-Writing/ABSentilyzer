# Title             : forms.py
# Author            : Keith Barrientos
#                     Afrahly Afable
# Date Written      : October 14, 2021
# Date Revised      : December 13, 2021
# Purpose           : Declare all the fields used in the form and provide 
#                     a customized validation



from django import forms
from pandas import *
import os

class InputForm(forms.Form):

    #SingleInputType
    text = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'rows':7, 
                'cols':0, 
                'id': 'textAreaWidget', 
                'placeholder': 'Insert text here...', 
                'maxlength': '300', 
                'spellcheck':'true'
            }
        ),
        label="", #removes the label of the text area
        required=False
    ) 

    #FileUploadInputType
    csv = forms.FileField(
        widget=forms.ClearableFileInput(
            attrs={
                'class': 'form-control',
                'id':'FileUploadCSV',
                'value' : 'Upload CSV',
                'multiple': False, 
                'accept':'.csv'
            }
        ),
        label="", #removes the label of the file upload button
        required=False
    ) 

    # customized form validation
    def clean(self):
        cleaned_data = super(InputForm, self).clean()
        text = cleaned_data.get('text')
        csv = cleaned_data.get('csv')
        
        self.isCSV = False 
        self.isText = False
        self.isNotLarge = False
        self.isNotInput = False
        self.inputText = text
        
        print(text)

        if csv:
            # file type validation
            ext = os.path.splitext(csv.name)[1] # returns path+filename
            valid_filetype = ['.csv']
            
            if ext.lower() in valid_filetype:
                try:
                    #read uploaded csv
                    data = read_csv(csv)
                except:
                    #read uploaded csv
                    data = read_csv(csv, encoding='cp1252')

                # the system has a note that the column the user wants to classify must be renamed as "input"
                # convert "input column into list"
                cols = [col for col in data.columns]
                if 'input' in cols:
                    self.included_col = data['input'].tolist()
                    # check no. of rows
                    self.check_rows = (len(self.included_col))

                    # number of rows validation
                    if self.check_rows <= 499:
                        # proceed to backend
                        self.isCSV = True
                        print(csv.name) #prints the file name in the console
                        print("proceed to backend")
                    else:
                        # else if row is greater than 500, raise csv_rowError validation then reload page
                        self.isNotLarge = True
                        raise forms.ValidationError('large message here')
                else:
                    self.isNotInput = True
                    raise forms.ValidationError('Columns no input')
            else:
                # else if filetype is not supported, raise filetype_error validation then reload page
                raise forms.ValidationError('filetypemessagehere')
        elif text != '':
            self.isText = True
        else:
            raise forms.ValidationError('You have to write or upload something!')

    def get_text(self):
        return self.inputText

    def get_csv(self):
        return self.included_col, self.check_rows
    
    def check_csv(self):
        return self.isCSV
    
    def check_text(self):
        return self.isText
