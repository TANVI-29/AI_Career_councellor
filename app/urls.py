from django.contrib import admin
from django.urls import path, include
from app import views
urlpatterns = [
    path("", views.home, name='home'),
    path("questions/", views.questions, name='questions'),
]
