"""Lưu trữ object trên MinIO: key (``paths``), đọc / ghi (``objects``), quarantine (``quarantine``).

Bucket ``raw`` là lớp 0, nằm ngoài Iceberg và chỉ crawler ghi vào: crawler đẩy file lên ``raw`` rồi gửi
message có key vào Kafka → Bronze (Iceberg: metadata + key) → Silver → Gold (PLAN mục 2, 4.2).
"""
