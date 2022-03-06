"""
    Title               : urls.py
    Author              : Keith Barrientos
    System Design       : Web user-interfaces
    Date Written        : October 05, 2021
    Date Revised        : December 9, 2021
    Purpose             : Maps the routes and paths in the app
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include('ensemble_analyzer.apps.public.urls'))
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
