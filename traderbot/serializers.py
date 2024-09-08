from rest_framework.serializers import ModelSerializer
from .models import Market, Trade, Signal



class MarketSerializer(ModelSerializer):
    class Meta:
        model = Market
        fields = '__all__'


class TradeSerializer(ModelSerializer):
    class Meta:
        model = Trade
        fields = '__all__'


class SignalSerializer(ModelSerializer):
    class Meta:
        model = Signal
        fields = '__all__'