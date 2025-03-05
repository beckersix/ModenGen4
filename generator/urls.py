"""
URL configuration for the generator app.

This file redirects to the actual URL configuration in the api package.
"""

from django.urls import path, include

# URL patterns
urlpatterns = [
    # Include all URLs from the api package
    path('', include('generator.api.urls')),
]
