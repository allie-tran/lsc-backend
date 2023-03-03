"""project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from django.contrib import admin
from . import views
import os
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenObtainSlidingView
from .auth_model import CustomTokenObtainPairView

os.environ['KMP_DUPLICATE_LIB_OK']='True'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('cross-server-auth', views.cross_server_auth),
    path('auth', CustomTokenObtainPairView.as_view()),
    path('auth/refresh', CustomTokenObtainPairView.as_view()),
    path('', include('images.urls'))
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
