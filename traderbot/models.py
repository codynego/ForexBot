from django.db import models



class BaseModel(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Market(BaseModel):
    name = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    time_frame = models.CharField(max_length=255)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.FloatField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = "Market"
        verbose_name_plural = "Markets"


class Trade(BaseModel):
    market = models.ForeignKey(Market, on_delete=models.CASCADE)
    entry_price = models.FloatField()
    exit_price = models.FloatField()
    stop_loss = models.FloatField()
    take_profit = models.FloatField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.market.name
    
    class Meta:
        verbose_name = "Trade"
        verbose_name_plural = "Trades"

class Signal(BaseModel):
    SIGNAL_CHOICES = (
        ('BUY', 'BUY'),
        ('SELL', 'SELL'),
        ('HOLD', 'HOLD'),
    )

    symbol = models.CharField(max_length=255, null=True)
    price = models.FloatField(null=True)
    type = models.CharField(max_length=255, choices=SIGNAL_CHOICES, null=True)
    strength = models.FloatField(null=True)
    is_active = models.BooleanField(default=True, null=True)

    def __str__(self):
        return self.symbol
    
    class Meta:
        verbose_name = "Signal"
        verbose_name_plural = "Signals"

class Indicator(BaseModel):
    market = models.ForeignKey(Market, on_delete=models.CASCADE)
    rsi = models.FloatField()
    macd = models.FloatField()
    bollinger_bands = models.FloatField()
    moving_average = models.FloatField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.market.name
    
    class Meta:
        verbose_name = "Indicator"
        verbose_name_plural = "Indicators"



