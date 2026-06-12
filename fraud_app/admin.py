from django.contrib import admin
from .models import Transaction, SecurityAuditLog


@admin.register(SecurityAuditLog)
class SecurityAuditLogAdmin(admin.ModelAdmin):
    list_display  = ('id', 'amount', 'anomaly_score', 'is_false_positive', 'timestamp')
    list_filter   = ('is_false_positive',)
    ordering      = ('-timestamp',)
    actions       = ['delete_selected']


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ('id', 'amount', 'status', 'timestamp')
    ordering     = ('-timestamp',)
