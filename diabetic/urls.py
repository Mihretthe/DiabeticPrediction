from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('predict/', views.predict_diabetes, name='predict'),
]
