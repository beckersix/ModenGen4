"""
URL configuration for the generator app.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from . import text_to_3d_views
from . import train_views
from . import module_status_views
from .views import datasets_view
from generator.api import image_generation_views

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
    
    # Dataset API endpoints
    path('api/datasets/shapenet/', views.get_shapenet_categories, name='get_shapenet_categories'),
    path('api/datasets/shapenet/download/', views.download_shapenet_category, name='download_shapenet_category'),
    path('api/datasets/shapenet/download-all/', views.download_all_shapenet, name='download_all_shapenet'),
    path('api/datasets/shapenet/download-status', views.get_shapenet_download_status, name='get_shapenet_download_status'),
    path('api/datasets/objectnet3d/categories', views.get_objectnet3d_categories, name='get_objectnet3d_categories'),
    path('api/datasets/objectnet3d/download', views.download_objectnet3d_category, name='download_objectnet3d_category'),
    path('api/datasets/objectnet3d/download-toolbox', views.download_objectnet3d_toolbox, name='download_objectnet3d_toolbox'),
    path('api/datasets/objectnet3d/download-all', views.download_all_objectnet3d, name='download_all_objectnet3d'),
    path('api/datasets/objectnet3d/download-status', views.get_objectnet3d_download_status, name='get_objectnet3d_download_status'),
    path('api/datasets/custom/', views.get_custom_datasets, name='custom_datasets'),
    path('api/datasets/combined/', views.get_combined_datasets, name='combined_datasets'),
    
    # Text-to-3D API endpoints
    path('api/text-to-3d/generate/', text_to_3d_views.generate_model_from_text, name='generate_model_from_text'),
    path('api/text-to-3d/status/<uuid:model_id>/', text_to_3d_views.model_generation_status, name='model_generation_status'),
    path('api/text-to-3d/available-models/', text_to_3d_views.list_available_models, name='list_available_models'),
    path('api/text-to-3d/model/<uuid:model_id>/', text_to_3d_views.model_detail, name='text_to_3d_model_detail'),
    path('api/text-to-3d/model/<uuid:model_id>/details/', text_to_3d_views.get_model_details, name='get_model_details'),
    path('api/text-to-3d/history/', text_to_3d_views.get_model_history, name='get_model_history'),
    
    # Training API endpoints
    path('api/train/start/', train_views.start_training, name='start_training'),
    path('api/train/status/<uuid:model_id>/', train_views.training_status, name='training_status'),
    path('api/train/stop/<uuid:model_id>/', train_views.stop_training, name='stop_training'),
    path('api/train/models/', train_views.list_trained_models, name='list_trained_models'),
    path('api/train/set-default/<uuid:model_id>/', train_views.set_default_model, name='set_default_model'),
    
    # Module status endpoint
    path('api/module-status/', module_status_views.module_status, name='module_status'),
    path('datasets/', datasets_view, name='datasets'),
    path('api/generate-multiview', image_generation_views.generate_multiview_images, name='generate_multiview'),
]
