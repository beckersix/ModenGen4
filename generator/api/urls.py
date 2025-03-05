"""
URL configuration for the generator app.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from . import text_to_3d_views
from . import train_views

# Create a router for REST API
router = DefaultRouter()
router.register(r'models', views.GeneratedModelViewSet)
router.register(r'settings', views.GenerationSettingViewSet)

# URL patterns
urlpatterns = [
    # Web views
    path('', views.TextTo3DView.as_view(), name='home'),
    path('models/', views.ModelListView.as_view(), name='model_list'),
    path('models/<uuid:pk>/', views.ModelDetailView.as_view(), name='model_detail'),
    path('train/', views.TrainView.as_view(), name='train'),
    
    # API endpoints
    path('api/', include(router.urls)),
    path('api/generate/', views.generate_model, name='generate_model'),
    
    # Text-to-3D API endpoints
    path('api/text-to-3d/generate/', text_to_3d_views.generate_model_from_text, name='generate_model_from_text'),
    path('api/text-to-3d/status/<uuid:model_id>/', text_to_3d_views.model_generation_status, name='model_generation_status'),
    path('api/text-to-3d/available-models/', text_to_3d_views.list_available_models, name='list_available_models'),
    path('api/text-to-3d/model/<uuid:model_id>/', text_to_3d_views.model_detail, name='text_to_3d_model_detail'),
    path('api/text-to-3d/model/<uuid:model_id>/details/', text_to_3d_views.get_model_details, name='get_model_details'),
    
    # Training API endpoints
    path('api/train/start/', train_views.start_training, name='start_training'),
    path('api/train/status/<uuid:model_id>/', train_views.training_status, name='training_status'),
    path('api/train/stop/<uuid:model_id>/', train_views.stop_training, name='stop_training'),
    path('api/train/models/', train_views.list_trained_models, name='list_trained_models'),
    path('api/train/set-default/<uuid:model_id>/', train_views.set_default_model, name='set_default_model'),
]
