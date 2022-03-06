# Author            : Keith Barrientos
# Calling Sequence  : N/A
# Date Written      : October 05, 2021
# Date Revised      : December 9, 2021
# Purpose           : Store all the web addresses for the website

from django.urls import path
from . import views

app_name="public"
urlpatterns = [
    path("", views.index, name="index"),
    path("Features", views.how, name="features"),
    path("About", views.about, name="about"),
]
