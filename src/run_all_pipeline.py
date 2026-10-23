import subprocess
import sys
import os

scripts = [
    ("Kiểm tra môi trường", "src/check_environment.py"),
    ("Bước 1: Tích hợp dữ liệu", "src/step_01_data_integration.py"),
    ("Bước 2: EDA & Làm sạch dữ liệu", "src/step_02_eda_cleaning.py"),
    ("Bước 3: Biến đổi & Tương quan", "src/step_03_transformation_correlation.py"),
    ("Bước 4: Mô hình hóa OLS & Stepwise", "src/step_04_ols_stepwise_modeling.py"),
    ("Bước 5: Kiểm định 5 giả định OLS", "src/step_05_assumption_tests.py"),
    ("Bước 6: Đánh giá & Dự báo mô hình", "src/step_06_evaluation_prediction.py"),
    ("Bước 7: Gom cụm K-Means thị trường", "src/step_07_clustering_mining.py"),
    ("Bước 8: Khởi tạo Master HTML Dashboard", "src/generate_dashboard.py")
]

print("=" * 85)
print("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE PHÂN TÍCH GIÁ DẦU BRENT & WTI (2021 - 2026)")
print("=" * 85)

for ten_buoc, duong_dan in scripts:
    print(f"\n>>> ĐANG THỰC THI: {ten_buoc} ({duong_dan})...")
    res = subprocess.run([sys.executable, duong_dan])
    if res.returncode != 0:
        print(f"\n[LỖI NGHIÊM TRỌNG] {ten_buoc} thất bại với mã lỗi {res.returncode}. Dừng pipeline!")
        sys.exit(res.returncode)

print("\n" + "=" * 85)
print("CHÚC MỪNG: TOÀN BỘ 7 BƯỚC VÀ MASTER DASHBOARD ĐÃ HOÀN TẤT 100%!")
print("Bạn có thể mở tệp 'index.html' hoặc 'reports/oil_analysis_dashboard.html' trên trình duyệt để tương tác.")
print("=" * 85)
