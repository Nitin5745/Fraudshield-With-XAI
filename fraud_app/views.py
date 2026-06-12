from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.db import models
from django.db.models import Count
from django.db.models.functions import TruncHour

import logging
from datetime import timedelta

from .models import Transaction, SecurityAuditLog
from .services import predict_fraud, trigger_shap_background

logger = logging.getLogger(__name__)


@api_view(['POST'])
def transaction_check_view(request):

    data = request.data

    try:

        result = predict_fraud(data)

    except Exception as e:

        logger.critical(f'CRITICAL: AI_SERVICE_DOWN - {str(e)}')

        tx = Transaction.objects.create(
            amount=data.get('amount', 0),
            status='Approved',
            **{f'V{i}': data.get(f'V{i}', 0.0) for i in range(1, 29)}
        )

        return Response(
            {
                'status': 'approved',
                'reason': 'fail-open'
            },
            status=200
        )

    if result['is_fraud']:

        log = SecurityAuditLog.objects.create(
            amount           = data.get('amount', 0),
            anomaly_score    = result['risk_score'],
            is_vpn           = bool(data.get('vpn',      False)),
            is_international = data.get('country', 'india') == 'international',
            is_new_device    = data.get('device',  'known') == 'new',
            is_highrisk      = bool(data.get('highrisk', False)),
            is_high_amount   = float(data.get('amount', 0)) > 50000,
            **{f'V{i}': data.get(f'V{i}', 0.0) for i in range(1, 29)}
        )

        trigger_shap_background(result['features'], log.id)

        return Response(
            {
                'status': 'blocked',
                'message': 'Suspicious transaction detected',
                'risk_score': result['risk_score']
            },
            status=403
        )

    tx = Transaction.objects.create(
        amount=data.get('amount', 0),
        status='Approved',
        anomaly_score=result['risk_score'],
        **{f'V{i}': data.get(f'V{i}', 0.0) for i in range(1, 29)}
    )

    return Response(
        {
            'status': 'approved',
            'transaction_id': tx.id
        },
        status=200
    )


@login_required(login_url='login')
def dashboard_view(request):

    # Merge approved + blocked into one recent list, newest first
    approved = list(
        Transaction.objects.values(
            'id', 'amount', 'anomaly_score', 'timestamp'
        ).annotate(
            entry_type=models.Value('approved', output_field=models.CharField())
        ).order_by('-timestamp')[:50]
    )

    blocked = list(
        SecurityAuditLog.objects.values(
            'id', 'amount', 'anomaly_score', 'timestamp'
        ).annotate(
            entry_type=models.Value('blocked', output_field=models.CharField())
        ).order_by('-timestamp')[:50]
    )

    # Merge, sort by timestamp, take latest 10, assign sequential display numbers
    all_logs = sorted(
        approved + blocked,
        key=lambda x: x['timestamp'],
        reverse=True
    )[:10]

    # Add display_num so the table shows #1, #2, #3... cleanly
    for i, entry in enumerate(all_logs, start=1):
        entry['display_num'] = i

    total_transactions = Transaction.objects.count() + SecurityAuditLog.objects.count()
    fraud_cases        = SecurityAuditLog.objects.count()
    fraud_rate         = round((fraud_cases / total_transactions) * 100, 2) if total_transactions else 0

    return render(
        request,
        'dashboard.html',
        {
            'logs':               all_logs,
            'total_transactions': total_transactions,
            'fraud_cases':        fraud_cases,
            'fraud_rate':         fraud_rate,
        }
    )


@login_required(login_url='login')
def simulator_view(request):
    return render(request, 'simulator.html')


@api_view(['GET'])
def get_shap_values(request, log_id):

    log = SecurityAuditLog.objects.get(id=log_id)

    return Response({
        'features': [f'V{i}' for i in range(1, 29)],
        'values': log.shap_values or [],
        'ready': log.shap_values is not None,
    })


@api_view(['GET'])
def live_stats(request):
    """Return last-12-hour hourly bucket data for the dashboard charts."""
    now   = timezone.now()
    since = now - timedelta(hours=12)

    # ── Hourly approved transactions ──────────────────────────────────
    approved_qs = (
        Transaction.objects
        .filter(timestamp__gte=since)
        .annotate(hour=TruncHour('timestamp'))
        .values('hour')
        .annotate(count=Count('id'))
        .order_by('hour')
    )

    # ── Hourly blocked (fraud) transactions ───────────────────────────
    blocked_qs = (
        SecurityAuditLog.objects
        .filter(timestamp__gte=since)
        .annotate(hour=TruncHour('timestamp'))
        .values('hour')
        .annotate(count=Count('id'))
        .order_by('hour')
    )

    # Build a complete 12-hour label list
    labels, approved_data, blocked_data = [], [], []
    approved_map = {row['hour']: row['count'] for row in approved_qs}
    blocked_map  = {row['hour']: row['count'] for row in blocked_qs}

    for h in range(12):
        bucket = (now - timedelta(hours=11 - h)).replace(minute=0, second=0, microsecond=0)
        label  = bucket.strftime('%H:00')
        labels.append(label)
        approved_data.append(approved_map.get(bucket, 0))
        blocked_data.append(blocked_map.get(bucket, 0))

    # ── Pie: real flag counts from all blocked logs (all time) ─────────
    qs = SecurityAuditLog.objects
    pie_vpn    = qs.filter(is_vpn=True).count()
    pie_intl   = qs.filter(is_international=True).count()
    pie_newdev = qs.filter(is_new_device=True).count()
    pie_hi_amt = qs.filter(is_high_amount=True).count()

    total_tx    = Transaction.objects.count() + SecurityAuditLog.objects.count()
    fraud_cases = SecurityAuditLog.objects.count()
    fraud_rate  = round((fraud_cases / total_tx) * 100, 2) if total_tx else 0

    return Response({
        'labels':        labels,
        'approved':      approved_data,
        'blocked':       blocked_data,
        'total_tx':      total_tx,
        'fraud_cases':   fraud_cases,
        'fraud_rate':    fraud_rate,
        'pie': {
            'international': pie_intl,
            'new_device':    pie_newdev,
            'vpn':           pie_vpn,
            'high_amount':   pie_hi_amt,
        },
    })


@api_view(['POST'])
def mark_false_positive(request, log_id):

    log = SecurityAuditLog.objects.get(id=log_id)

    log.is_false_positive = True

    log.save()

    return Response({'status': 'updated'})


def login_view(request):

    if request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:
            login(request, user)
            return redirect('dashboard')

        else:
            messages.error(
                request,
                'Invalid username or password'
            )

    return render(request, 'login.html')


def logout_view(request):

    logout(request)

    return redirect('login')