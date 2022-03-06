<!-- 
    This module is intended for other people to read for them to know the important
    information about the project 

    # Author            : Group 4
    # Calling Sequence  : N/A
    # Date Written      : October 5, 2021
    # Date Revised      : December 9, 2021
    # Purpose           : Give instructions and information about the project
-->

# FExBE_Ensemble-Classifier-SVM-MNB-ATPC
-----------------------------
##### On running the server: 
   ```
   python manage.py runserver
   ```
-----------------------------
##### Installations
* Installing pip
   * You can make sure that pip is up-to-date by running:
      ```
      py -m pip install --upgrade pip
      py -m pip --version
      ```
* Installing virtualenv
   ```
   py -m pip install --user virtualenv
   ```
* Creating a virtual environment
   ```
   py -m venv env
   ```
* Activating a virtual environment
   ```
   .\env\Scripts\activate
   ```
* Install Django
   ```
   pip install Django
   ```
* Shell
   ```
   python
   >>> import Django
   >>> ^Z
   ```
* Create the boilerplate in the project 
   * these are sections of code that are repeated in multiple places with little to no variation)
      ```
      django-admin startproject ensemble_analyzer
      ```
* On Creating an app
   ```
   python manage.py startapp public projectfoldername/apps/public
   ```
##### Additional instructions:
  * You can make sure that django-heroku is installed by running:
      ```
      pip install django-heroku
      ```
-----------------------------
##### Django Python Scripts Basic Descriptions

* urls.py
   * path('') <-- root folder
* settings.py
   * a file which defines bunch of different settings in a django application
   * tell django which folder to look in when we reference our templates file
* views.py
   * this is where all views for a Django application are placed
   * takes in a request and returns a response
* forms.py
   * it is where the django documentation recommends you place all the forms code
   * will keep the code easily maintainable.
* manage.py
   * django's command-line utility for administrative tasks.
-----------------------------
##### Palettes of the Public App:
   * color:#F8F1F1;
   * color:#11698E; /*primary color*/
   * color:#19456B;
   * color:#16C79A;
   * color: #2C3E50; /*navbar color*/
-----------------------------

