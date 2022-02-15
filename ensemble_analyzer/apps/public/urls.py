'''
  This module serves as the address book of public app
'''

# Author            : Group 4
# Calling Sequence  : N/A
# Date Written      : October 05, 2021
# Date Revised      : December 9, 2021
# Purpose           : Store all the web addresses for the website

from django.urls import path
from . import views

app_name="public"
urlpatterns = [
    path("", views.index, name="index"),
    path("HowToUse", views.how, name="how"),
    path("About", views.about, name="about"),
]