from django.contrib import admin
from .models import Signal, Trade,  Indicator

# Register your models here.

admin.site.register(Signal)
admin.site.register(Trade)
admin.site.register(Indicator)
