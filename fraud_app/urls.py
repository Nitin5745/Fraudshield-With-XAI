from django.urls import path
from django.shortcuts import redirect

from .views import (
    transaction_check_view,
    dashboard_view,
    get_shap_values,
    mark_false_positive,
    login_view,
    logout_view,
    simulator_view,
    live_stats,
)

urlpatterns = [

    path('', lambda request: redirect('login')),

    path(
        'process-payment/',
        transaction_check_view
    ),

    path(
        'dashboard/',
        dashboard_view,
        name='dashboard'
    ),

    path(
        'login/',
        login_view,
        name='login'
    ),

    path(
        'logout/',
        logout_view,
        name='logout'
    ),

    path(
        'simulator/',
        simulator_view,
        name='simulator'
    ),

    path(
        'shap/<int:log_id>/',
        get_shap_values
    ),

    path(
        'mark-false-positive/<int:log_id>/',
        mark_false_positive
    ),

    path(
        'live-stats/',
        live_stats,
        name='live_stats'
    ),
]