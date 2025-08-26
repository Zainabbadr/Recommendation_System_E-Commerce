from django.urls import path
from . import views
app_name = 'recommendations'
urlpatterns = [
    path('', views.login_view, name='login'),
    path('index/', views.index, name='index'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('health/', views.health_check, name='health'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.customer_dashboard_view, name='customer_dashboard'),
    path('get_chat_history/', views.get_chat_history, name='get_chat_history'),
    path('save_chat_messages/', views.save_chat_messages, name='save_chat_messages'),
    path('delete_chat/', views.delete_chat, name='delete_chat'),
    path('api/purchase-history/', views.get_purchase_history, name='get_purchase_history'),
]