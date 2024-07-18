from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate/', views.generate_report, name='generate_report'),
    path('search/', views.search_chroma, name='search'),
]