from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'markets', views.MarketViewSet)
router.register(r'trades', views.TradeViewSet)
router.register(r'signals', views.SignalViewSet)

urlpatterns = [
    path('', include(router.urls)),
]