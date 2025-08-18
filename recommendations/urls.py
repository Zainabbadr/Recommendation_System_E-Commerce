from django.urls import path
from . import views
app_name = 'recommendations'
urlpatterns = [
    path('', views.login_view, name='login'),
    path('index/', views.index, name='index'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('health/', views.health_check, name='health'),
    path('logout/', views.logout_view, name='logout'),
]
