'''
    This module is intended for the declaration of all the fields that is used in the form
'''

# Author            : Group 4
# Calling Sequence  : InputForm(forms.Form)
# Date Written      : October 14, 2021
# Date Revised      : December 13, 2021
# Purpose           : Declare all the fields used in the form and provide a customized validation

from django import forms

class InputForm(forms.Form):

    #SimpleTextInputType
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
        if not text and not csv:
            raise forms.ValidationError('You have to write or upload something!')