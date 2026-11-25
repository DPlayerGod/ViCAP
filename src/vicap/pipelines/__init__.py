"""Điểm vào của từng bước, DAG ở ``dags/`` gọi vào đây (PLAN mục 9.2, 9.5).

Mỗi bước có một hàm ``run_<bước>(batch_id)``: đọc config, gọi logic của bước (nằm trong ``bronze/``,
``silver/``, ``features/``, ``model/``, ``gold/``), kiểm định rồi ghi kết quả. Task DAG chỉ gọi đúng một
hàm như vậy, không chép lại logic.

``run_daily(batch_id)`` nối các bước crawl → bronze → silver_clean → asr → embed → predict → profile →
load_clickhouse → quality_check, giống ``pipelines/`` của HOSE, để chạy không cần Airflow và test đầu–cuối.
"""
