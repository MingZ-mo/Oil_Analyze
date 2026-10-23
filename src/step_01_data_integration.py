import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def kiem_ke_tap_tin_tho(thu_muc_data="data"):
    danh_sach_file = sorted([f for f in os.listdir(thu_muc_data) if f.endswith(".csv")])
    
    print("=" * 85)
    print("TASK 1.1: KIỂM KÊ CẤU TRÚC CHI TIẾT CÁC TỆP DỮ LIỆU THÔ (DATA AUDIT)")
    print("=" * 85)
    for i, fn in enumerate(danh_sach_file, 1):
        duong_dan = os.path.join(thu_muc_data, fn)
        kich_thuoc = os.path.getsize(duong_dan)
        df_temp = pd.read_csv(duong_dan)
        print(f"[{i:02d}] {fn:<25} | Kích thước: {kich_thuoc:>8,} bytes | Số dòng: {len(df_temp):>5} | Cột: {list(df_temp.columns)}")
    print("-" * 85)

def dong_bo_va_tich_hop_du_lieu_dau():
    print("=" * 85)
    print("TASK 1.2: ĐỒNG BỘ CHUỖI THỜI GIAN & TÍCH HỢP DỮ LIỆU BRENT VÀ WTI (TASKS 1.2 - 1.5)")
    print("=" * 85)

    duong_dan_brent = os.path.join("data", "DCOILBRENTEU.csv")
    duong_dan_wti = os.path.join("data", "DCOILWTICO.csv")

    assert os.path.exists(duong_dan_brent), f"Lỗi: Không tìm thấy file {duong_dan_brent}"
    assert os.path.exists(duong_dan_wti), f"Lỗi: Không tìm thấy file {duong_dan_wti}"

    # 1. Đọc dữ liệu thô
    df_brent = pd.read_csv(duong_dan_brent)
    df_wti = pd.read_csv(duong_dan_wti)

    df_brent = df_brent.rename(columns={"observation_date": "Date", "DCOILBRENTEU": "Brent_Price"})
    df_wti = df_wti.rename(columns={"observation_date": "Date", "DCOILWTICO": "WTI_Price"})

    # 2. Xử lý kiểu dữ liệu số & ngày tháng
    df_brent["Brent_Price"] = pd.to_numeric(df_brent["Brent_Price"], errors="coerce")
    df_wti["WTI_Price"] = pd.to_numeric(df_wti["WTI_Price"], errors="coerce")

    # 3. Ghép nối theo ngày
    df_merged = pd.merge(df_brent, df_wti, on="Date", how="outer")
    df_merged["Date_Parsed"] = pd.to_datetime(df_merged["Date"])
    df_merged = df_merged.sort_values(by="Date_Parsed").reset_index(drop=True)

    # 4. Tạo các biến kinh tế lượng & tài chính
    # Chênh lệch giá Brent - WTI (Spread)
    df_merged["Spread_Brent_WTI"] = np.round(df_merged["Brent_Price"] - df_merged["WTI_Price"], 4)

    # Biến trễ Lag 1
    df_merged["WTI_Price_Lag1"] = df_merged["WTI_Price"].shift(1)
    df_merged["Brent_Price_Lag1"] = df_merged["Brent_Price"].shift(1)

    # Biến thời gian & mùa vụ
    df_merged["Year"] = df_merged["Date_Parsed"].dt.year
    df_merged["Month"] = df_merged["Date_Parsed"].dt.month
    df_merged["Quarter"] = df_merged["Date_Parsed"].dt.quarter
    season_map = {1: "Q1_Winter", 2: "Q2_Spring", 3: "Q3_Summer", 4: "Q4_Autumn"}
    df_merged["Season"] = df_merged["Quarter"].map(season_map)
    df_merged["DayOfWeek"] = df_merged["Date_Parsed"].dt.day_name()

    # Sắp xếp cấu trúc cột
    cot_chuan = [
        "Date",
        "WTI_Price",
        "WTI_Price_Lag1",
        "Brent_Price",
        "Brent_Price_Lag1",
        "Spread_Brent_WTI",
        "Year",
        "Month",
        "Quarter",
        "Season",
        "DayOfWeek"
    ]
    df_final = df_merged[cot_chuan].copy()

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    duong_dan_xuat = os.path.join("data", "processed", "oil_merged_raw.csv")
    df_final.to_csv(duong_dan_xuat, index=False, encoding="utf-8")

    print(f"- Đã tích hợp thành công {len(df_final)} bản ghi chuỗi thời gian.")
    print(f"- Khoảng thời gian: Từ {df_final['Date'].iloc[0]} đến {df_final['Date'].iloc[-1]}")
    print(f"- Tệp xuất: {duong_dan_xuat}")
    print("=" * 85)
    return df_final

def main():
    kiem_ke_tap_tin_tho()
    dong_bo_va_tich_hop_du_lieu_dau()
    print("[THÀNH CÔNG] HOÀN TẤT BƯỚC 1: TÍCH HỢP VÀ ĐỒNG BỘ DỮ LIỆU DẦU THÔ.")

if __name__ == "__main__":
    main()
