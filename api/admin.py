from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'outcome', 'goal_diff')
    readonly_fields = ('timestamp',)
