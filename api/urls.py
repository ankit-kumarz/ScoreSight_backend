from django.urls import path
from .views import (
    HealthView, TeamsView, PredictV2View, DebugInputView,
    SignupView, LoginView, UserStatsView
)

urlpatterns = [
    path('health', HealthView.as_view(), name='health'),
    path('teams', TeamsView.as_view(), name='teams'),
    path('predict_v2', PredictV2View.as_view(), name='predict_v2'),
    path('debug_input', DebugInputView.as_view(), name='debug_input'),
    path('signup', SignupView.as_view(), name='signup'),
    path('login', LoginView.as_view(), name='login'),
    path('user/stats', UserStatsView.as_view(), name='user_stats'),
]
