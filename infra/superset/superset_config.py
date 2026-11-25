"""Cấu hình Superset: không có giá trị mặc định cho bí mật (PLAN mục 9.2)."""

import os

SECRET_KEY = os.environ["SUPERSET_SECRET_KEY"]
SQLALCHEMY_DATABASE_URI = os.environ["SUPERSET_DATABASE_URI"]

# Metadata DB nhỏ, chỉ một người dùng; không cần Redis / Celery.
CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300}
DATA_CACHE_CONFIG = CACHE_CONFIG
FEATURE_FLAGS = {"DASHBOARD_NATIVE_FILTERS": True}
WTF_CSRF_ENABLED = True
TALISMAN_ENABLED = False  # chỉ chạy local qua http
