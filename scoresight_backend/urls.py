from django.urls import path, include
from django.http import JsonResponse

def root_view(request):
    return JsonResponse({
        'status': 'online',
        'service': 'ScoreSight Backend API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'teams': '/api/teams',
            'predict': '/api/predict_v2',
            'signup': '/api/signup',
            'login': '/api/login',
            'user_stats': '/api/user/stats'
        }
    })

urlpatterns = [
    path('', root_view, name='root'),
    path('api/', include('api.urls')),
]
