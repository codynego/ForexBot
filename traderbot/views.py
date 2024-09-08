from django.shortcuts import render
from rest_framework import viewsets
from .models import Market, Trade, Signal
from .serializers import MarketSerializer, TradeSerializer, SignalSerializer


# Create your views here.

class MarketViewSet(viewsets.ModelViewSet):
    queryset = Market.objects.all()
    serializer_class = MarketSerializer


class TradeViewSet(viewsets.ModelViewSet):
    queryset = Trade.objects.all()
    serializer_class = TradeSerializer

class SignalViewSet(viewsets.ModelViewSet):
    queryset = Signal.objects.all()
    serializer_class = SignalSerializer
