import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def thong_ke_mo_ta_ban_dau(df, cac_cot_dinh_luong):
    print("=" * 85)
    print("TASK 2.1: BẢNG THỐNG KÊ MÔ TẢ TẬP DỮ LIỆU BAN ĐẦU (RAW EDA STATISTICAL SUMMARY)")
    print("=" * 85)
    
    bang_thong_ke = []
    for cot in cac_cot_dinh_luong:
        chuoi = df[cot].dropna()
        mode_series = chuoi.mode()
        bang_thong_ke.append({
            "Biến số": cot,
            "Số mẫu (N)": int(chuoi.count()),
            "Mean": round(chuoi.mean(), 2),
            "Std": round(chuoi.std(), 2),
            "Median": round(chuoi.median(), 2),
            "Mode": round(mode_series.iloc[0] if len(mode_series) > 0 else np.nan, 2),
            "Min": round(chuoi.min(), 2),
            "Q25 (25%)": round(chuoi.quantile(0.25), 2),
            "Q75 (75%)": round(chuoi.quantile(0.75), 2),
            "Max": round(chuoi.max(), 2)
        })
        
    df_thong_ke = pd.DataFrame(bang_thong_ke)
    print(df_thong_ke.to_string(index=False))
    return df_thong_ke

def xu_ly_trung_lap(df):
    print("\n" + "=" * 85)
    print("TASK 2.2: KIỂM TRA VÀ XỬ LÝ DỮ LIỆU TRÙNG LẶP (DUPLICATE REMOVAL)")
    print("=" * 85)
    
    so_dong_trung = df.duplicated(subset=["Date"]).sum()
    print(f"- Số bản ghi trùng lặp mốc thời gian 'Date': {so_dong_trung} dòng.")
    if so_dong_trung > 0:
        df_khong_trung = df.drop_duplicates(subset=["Date"], keep="first").copy()
    else:
        df_khong_trung = df.copy()
    return df_khong_trung

def xu_ly_gia_tri_khuyet(df, cac_cot_gia):
    print("\n" + "=" * 85)
    print("TASK 2.3: KIỂM TRA VÀ XỬ LÝ GIÁ TRỊ KHUYẾT (MISSING VALUES IMPUTATION)")
    print("=" * 85)
    
    tong_so_dong = len(df)
    for col in cac_cot_gia:
        so_khuyet = df[col].isnull().sum()
        ty_le = (so_khuyet / tong_so_dong) * 100
        print(f"- Cột '{col:<18}': {so_khuyet:>3} ô khuyết ({ty_le:.2f}%)")
        
    df_imputed = df.copy()
    # Với chuỗi thời gian tài chính, phương pháp Forward-Fill rồi Backward-Fill là chuẩn mực
    for col in cac_cot_gia:
        df_imputed[col] = df_imputed[col].ffill().bfill()
        
    # Cập nhật lại các biến phái sinh sau khi điền khuyết
    df_imputed["Spread_Brent_WTI"] = np.round(df_imputed["Brent_Price"] - df_imputed["WTI_Price"], 4)
    df_imputed["WTI_Price_Lag1"] = df_imputed["WTI_Price"].shift(1).bfill()
    df_imputed["Brent_Price_Lag1"] = df_imputed["Brent_Price"].shift(1).bfill()
    
    print("\n-> Kết quả sau khi điền khuyết (Forward-Fill):")
    print(f"- Tổng số ô khuyết còn lại: {df_imputed[cac_cot_gia].isnull().sum().sum()}")
    return df_imputed

def phat_hien_va_xu_ly_ngoai_lai_3sigma(df, cac_cot_dinh_luong):
    print("\n" + "=" * 85)
    print("TASK 2.4: PHÁT HIỆN VÀ XỬ LÝ NGOẠI LAI THEO QUY TẮC 3-SIGMA (3-SIGMA RULE)")
    print("=" * 85)
    
    df_cleaned = df.copy()
    nhat_ky_3sigma = []
    
    for cot in cac_cot_dinh_luong:
        mean_val = df_cleaned[cot].mean()
        std_val = df_cleaned[cot].std()
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        
        ngoai_lai = df_cleaned[(df_cleaned[cot] < lower_bound) | (df_cleaned[cot] > upper_bound)]
        so_luong_nl = len(ngoai_lai)
        
        nhat_ky_3sigma.append({
            "Biến số": cot,
            "Mean": round(mean_val, 2),
            "Std": round(std_val, 2),
            "Hàng rào dưới (-3σ)": round(lower_bound, 2),
            "Hàng rào trên (+3σ)": round(upper_bound, 2),
            "Số ngoại lai": so_luong_nl
        })
        # Winsorize / Capping tại 3-Sigma để tránh làm méo mó mô hình hồi quy
        df_cleaned[cot] = np.clip(df_cleaned[cot], lower_bound, upper_bound)
        
    df_3sigma_rep = pd.DataFrame(nhat_ky_3sigma)
    print(df_3sigma_rep.to_string(index=False))
    return df_cleaned, df_3sigma_rep

def truc_quan_hoa_eda(df_raw, df_cleaned, fig01_png, fig01_html, fig02_png, fig02_html):
    print("\n" + "=" * 85)
    print("TASK 2.5: TRỰC QUAN HÓA DỮ LIỆU EDA (XUẤT RA ẢNH VÀ FILE HTML TƯƠNG TÁC)")
    print("=" * 85)
    
    os.makedirs(os.path.dirname(fig01_png), exist_ok=True)
    
    # ---------------- 1. TẠO BIỂU ĐỒ CHUỖI THỜI GIAN PLOTLY (HTML) ----------------
    fig_ts = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Giá Dầu Thô Hàng Ngày: Brent vs WTI (2021 - 2026)", "Chênh Lệch Giá (Brent - WTI Spread) ($/Thùng)"),
        row_heights=[0.65, 0.35]
    )
    
    # Trace 1: Brent
    fig_ts.add_trace(
        go.Scatter(
            x=df_cleaned["Date"], y=df_cleaned["Brent_Price"],
            mode="lines", name="Dầu Brent (Châu Âu)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Ngày: %{x}<br>Brent: $%{y:.2f}/thùng<extra></extra>"
        ),
        row=1, col=1
    )
    # Trace 2: WTI
    fig_ts.add_trace(
        go.Scatter(
            x=df_cleaned["Date"], y=df_cleaned["WTI_Price"],
            mode="lines", name="Dầu WTI (Mỹ)",
            line=dict(color="#d62728", width=2),
            hovertemplate="Ngày: %{x}<br>WTI: $%{y:.2f}/thùng<extra></extra>"
        ),
        row=1, col=1
    )
    # Trace 3: Spread
    fig_ts.add_trace(
        go.Scatter(
            x=df_cleaned["Date"], y=df_cleaned["Spread_Brent_WTI"],
            mode="lines", name="Spread (Brent - WTI)",
            line=dict(color="#2ca02c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.15)",
            hovertemplate="Ngày: %{x}<br>Spread: $%{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    fig_ts.update_layout(
        title=dict(text="<b>TỔNG QUAN BIẾN ĐỘNG THỊ TRƯỜNG DẦU THÔ BRENT & WTI (2021 - 2026)</b>", font=dict(size=16)),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6 Tháng", step="month", stepmode="backward"),
                    dict(count=1, label="1 Năm", step="year", stepmode="backward"),
                    dict(count=3, label="3 Năm", step="year", stepmode="backward"),
                    dict(step="all", label="Toàn Bộ")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.08),
            type="date"
        ),
        height=750
    )
    fig_ts.write_html(fig01_html, include_plotlyjs="cdn")
    print(f"- Đã xuất biểu đồ tương tác HTML: {fig01_html}")

    # Vẽ và lưu bản tĩnh PNG fig01
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    dates = pd.to_datetime(df_cleaned["Date"])
    ax1.plot(dates, df_cleaned["Brent_Price"], label="Dầu Brent (Châu Âu)", color="#1f77b4", linewidth=1.8)
    ax1.plot(dates, df_cleaned["WTI_Price"], label="Dầu WTI (Mỹ)", color="#d62728", linewidth=1.8)
    ax1.set_title("BIẾN ĐỘNG GIÁ DẦU THÔ BRENT VÀ WTI (2021 - 2026)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("USD / Thùng", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)
    
    ax2.plot(dates, df_cleaned["Spread_Brent_WTI"], label="Spread (Brent - WTI)", color="#2ca02c", linewidth=1.4)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.fill_between(dates, df_cleaned["Spread_Brent_WTI"], 0, alpha=0.2, color="#2ca02c")
    ax2.set_title("CHÊNH LỆCH GIÁ (BRENT - WTI SPREAD)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Thời gian", fontsize=11)
    ax2.set_ylabel("USD", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(fig01_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất ảnh tĩnh PNG: {fig01_png}")

    # ---------------- 2. TẠO BIỂU ĐỒ BOXPLOT & PHÂN PHỐI PLOTLY (HTML) ----------------
    fig_box = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Phân phối Giá Dầu WTI", "Phân phối Giá Dầu Brent", "Phân phối Chênh Lệch Spread")
    )
    fig_box.add_trace(go.Box(y=df_cleaned["WTI_Price"], name="WTI Price", marker_color="#d62728", boxpoints="outliers"), row=1, col=1)
    fig_box.add_trace(go.Box(y=df_cleaned["Brent_Price"], name="Brent Price", marker_color="#1f77b4", boxpoints="outliers"), row=1, col=2)
    fig_box.add_trace(go.Box(y=df_cleaned["Spread_Brent_WTI"], name="Spread (Brent - WTI)", marker_color="#2ca02c", boxpoints="outliers"), row=1, col=3)
    
    fig_box.update_layout(
        title=dict(text="<b>PHÂN PHỐI VÀ HÀNG RÀO NGOẠI LAI (BOXPLOTS) CÁC BIẾN KINH TẾ LƯỢNG</b>", font=dict(size=15)),
        template="plotly_white",
        showlegend=False,
        height=500
    )
    fig_box.write_html(fig02_html, include_plotlyjs="cdn")
    print(f"- Đã xuất biểu đồ tương tác HTML: {fig02_html}")

    # Bản tĩnh PNG fig02
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    cols = ["WTI_Price", "Brent_Price", "Spread_Brent_WTI"]
    titles = ["Giá Dầu WTI", "Giá Dầu Brent", "Spread (Brent - WTI)"]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    
    for ax, col, tit, clr in zip(axes, cols, titles, colors):
        sns.boxplot(y=df_cleaned[col], ax=ax, color=clr, width=0.35)
        sns.stripplot(y=df_cleaned[col], ax=ax, color="black", alpha=0.15, jitter=0.2, size=3)
        ax.set_title(tit, fontsize=11, fontweight="bold")
        ax.set_ylabel("USD / Thùng")
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.suptitle("PHÂN PHỐI VÀ NGOẠI LAI CỦA CÁC BIẾN DẦU THÔ (BOXPLOT & STRIPPLOT)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig02_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất ảnh tĩnh PNG: {fig02_png}")

def main():
    file_raw = os.path.join("data", "processed", "oil_merged_raw.csv")
    file_cleaned = os.path.join("data", "processed", "oil_cleaned.csv")
    
    fig01_png = os.path.join("figures", "fig01_timeseries_overview.png")
    fig01_html = os.path.join("figures", "fig01_timeseries_overview.html")
    fig02_png = os.path.join("figures", "fig02_boxplots_distribution.png")
    fig02_html = os.path.join("figures", "fig02_boxplots_distribution.html")
    
    assert os.path.exists(file_raw), f"Lỗi: Không tìm thấy '{file_raw}'! Hãy chạy step 1 trước."
    df_raw = pd.read_csv(file_raw)
    
    cac_cot_dinh_luong = ["WTI_Price", "Brent_Price", "Spread_Brent_WTI", "WTI_Price_Lag1"]
    cac_cot_gia = ["WTI_Price", "Brent_Price"]
    
    df_stat_raw = thong_ke_mo_ta_ban_dau(df_raw, cac_cot_dinh_luong)
    df_no_dup = xu_ly_trung_lap(df_raw)
    df_imputed = xu_ly_gia_tri_khuyet(df_no_dup, cac_cot_gia)
    df_cleaned, df_3sigma = phat_hien_va_xu_ly_ngoai_lai_3sigma(df_imputed, cac_cot_dinh_luong)
    
    truc_quan_hoa_eda(df_raw, df_cleaned, fig01_png, fig01_html, fig02_png, fig02_html)
    
    df_cleaned.to_csv(file_cleaned, index=False, encoding="utf-8")
    
    assert df_cleaned.isnull().sum().sum() == 0, "Self-Test Lỗi: Vẫn còn ô trống (NaN)!"
    assert df_cleaned["Date"].duplicated().sum() == 0, "Self-Test Lỗi: Trùng mốc thời gian!"
    assert os.path.exists(fig01_html) and os.path.getsize(fig01_html) > 0, "Lỗi file HTML 01!"
    assert os.path.exists(fig02_html) and os.path.getsize(fig02_html) > 0, "Lỗi file HTML 02!"
    assert os.path.exists(file_cleaned) and os.path.getsize(file_cleaned) > 0, "Lỗi file cleaned rỗng!"
    
    print(f"\n[THÀNH CÔNG] Đã lưu file sạch: {file_cleaned} ({len(df_cleaned)} dòng).")
    print("SELF-TEST BƯỚC 2: PASS 100%!")

if __name__ == "__main__":
    main()
