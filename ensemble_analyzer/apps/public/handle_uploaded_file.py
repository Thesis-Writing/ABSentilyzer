import os
from django.core.files.storage import default_storage


def handle_uploaded_file(f):
    with open(default_storage.path('tmp/'+f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
            print(destination.write(chunk))