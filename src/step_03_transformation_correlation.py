import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def thuc_hien_roi_rac_hoa_va_label_encoding(df):
    print("=" * 85)
    print("TASK 3.1: RỜI RẠC HÓA DỮ LIỆU (pd.cut) & MÃ HÓA NHÃN (LabelEncoder)")
    print("=" * 85)

    df_trans = df.copy()
    nhan_danh_muc = ["Thap", "Trung_Binh", "Cao"]
    df_trans["Brent_Level"] = pd.cut(
        df_trans["Brent_Price"], 
        bins=3, 
        labels=nhan_danh_muc
    )
    
    le = LabelEncoder()
    df_trans["Brent_Level_Encoded"] = le.fit_transform(df_trans["Brent_Level"].astype(str))
    
    print("- Bảng phân bổ tần số các mức giá dầu Brent:")
    print(df_trans["Brent_Level"].value_counts())
    print("\n- Ánh xạ nhãn (Label Encoding):")
    for idx, cl in enumerate(le.classes_):
        print(f"  + Mức '{cl}': {idx}")
        
    return df_trans, le

def thuc_hien_one_hot_encoding(df):
    print("\n" + "=" * 85)
    print("TASK 3.2: MÃ HÓA BIẾN ĐỊNH DANH MÙA VỤ BẰNG ONE-HOT ENCODING (pd.get_dummies)")
    print("=" * 85)

    df_encoded = df.copy()
    df_encoded = pd.get_dummies(
        df_encoded, 
        columns=["Season"], 
        drop_first=True, 
        dtype=int
    )
    cols_season = [c for c in df_encoded.columns if c.startswith("Season_")]
    print(f"- Các cột One-Hot đã tạo cho Season (drop_first=True): {cols_season}")
    return df_encoded

def phan_tich_tuong_quan_pearson(df, cac_bien_doc_lap, bien_phu_thuoc):
    print("\n" + "=" * 85)
    print("TASK 3.3: PHÂN TÍCH TƯƠNG QUAN TUYẾN TÍNH PEARSON & KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ")
    print("=" * 85)

    y = df[bien_phu_thuoc]
    bang_tuong_quan = []

    for x_col in cac_bien_doc_lap:
        x = df[x_col]
        r_val, p_val = pearsonr(x, y)
        if abs(r_val) >= 0.8:
            muc_do = "Rất mạnh"
        elif abs(r_val) >= 0.6:
            muc_do = "Mạnh"
        elif abs(r_val) >= 0.4:
            muc_do = "Vừa phải"
        else:
            muc_do = "Yếu"
            
        y_nghia = "Có ý nghĩa (p < 0.05)" if p_val < 0.05 else "Không có ý nghĩa"
        bang_tuong_quan.append({
            "Biến độc lập (X)": x_col,
            "Hệ số tương quan (r)": round(r_val, 4),
            "p-value": f"{p_val:.4e}",
            "Mức độ tương quan": muc_do,
            "Ý nghĩa thống kê": y_nghia
        })

    df_corr_summary = pd.DataFrame(bang_tuong_quan)
    print(df_corr_summary.to_string(index=False))
    return df_corr_summary

def truc_quan_hoa_tuong_quan(df, cac_bien, fig03_png, fig03_html, fig04_png, fig04_html):
    print("\n" + "=" * 85)
    print("TASK 3.4: TRỰC QUAN HÓA MA TRẬN TƯƠNG QUAN & SCATTER PLOT (HTML & PNG)")
    print("=" * 85)

    corr_matrix = df[cac_bien].corr().round(3)

    # 1. HEATMAP TƯƠNG TÁC PLOTLY (HTML)
    fig_heat = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1, zmax=1,
        title="<b>MA TRẬN HỆ SỐ TƯƠNG QUAN PEARSON THỊ TRƯỜNG DẦU THÔ</b>"
    )
    fig_heat.update_layout(
        template="plotly_white",
        height=550,
        font=dict(size=12)
    )
    fig_heat.write_html(fig03_html, include_plotlyjs="cdn")
    print(f"- Đã xuất Heatmap tương tác HTML: {fig03_html}")

    # Bản tĩnh Heatmap PNG
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("MA TRẬN TƯƠNG QUAN PEARSON (BRENT, WTI & LAGS)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(fig03_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất Heatmap PNG: {fig03_png}")

    # 2. SCATTER PLOT BRENT VS WTI TƯƠNG TÁC PLOTLY (HTML)
    fig_scat = px.scatter(
        df,
        x="Brent_Price",
        y="WTI_Price",
        color="Spread_Brent_WTI",
        color_continuous_scale="Viridis",
        trendline="ols",
        hover_data=["Date", "Brent_Price", "WTI_Price", "Spread_Brent_WTI"],
        labels={"Brent_Price": "Giá Dầu Brent ($/thùng)", "WTI_Price": "Giá Dầu WTI ($/thùng)", "Spread_Brent_WTI": "Spread ($)"},
        title="<b>TƯƠNG QUAN TUYẾN TÍNH GIỮA DẦU BRENT VÀ DẦU WTI (KÈM ĐƯỜNG XU HƯỚNG OLS)</b>"
    )
    fig_scat.update_layout(template="plotly_white", height=600)
    fig_scat.write_html(fig04_html, include_plotlyjs="cdn")
    print(f"- Đã xuất Scatter Plot tương tác HTML: {fig04_html}")

    # Bản tĩnh Scatter Plot PNG
    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=df, x="Brent_Price", y="WTI_Price",
        scatter_kws={"alpha": 0.5, "color": "#1f77b4", "s": 25},
        line_kws={"color": "#d62728", "linewidth": 2}
    )
    plt.title("MỐI QUAN HỆ TUYẾN TÍNH: BRENT PRICE VS WTI PRICE", fontsize=13, fontweight="bold")
    plt.xlabel("Giá Dầu Brent (USD/thùng)", fontsize=11)
    plt.ylabel("Giá Dầu WTI (USD/thùng)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig04_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất Scatter Plot PNG: {fig04_png}")

def main():
    file_cleaned = os.path.join("data", "processed", "oil_cleaned.csv")
    file_transformed = os.path.join("data", "processed", "oil_transformed.csv")
    
    fig03_png = os.path.join("figures", "fig03_correlation_heatmap.png")
    fig03_html = os.path.join("figures", "fig03_correlation_heatmap.html")
    fig04_png = os.path.join("figures", "fig04_scatter_brent_wti.png")
    fig04_html = os.path.join("figures", "fig04_scatter_brent_wti.html")

    assert os.path.exists(file_cleaned), f"Lỗi: Không tìm thấy '{file_cleaned}'! Chạy step 2 trước."
    df = pd.read_csv(file_cleaned)

    # 1. Rời rạc hóa & Label Encoding
    df_trans, le = thuc_hien_roi_rac_hoa_va_label_encoding(df)

    # 2. One-Hot Encoding
    df_encoded = thuc_hien_one_hot_encoding(df_trans)

    # 3. Phân tích tương quan
    cac_bien_doc_lap = ["Brent_Price", "WTI_Price_Lag1", "Brent_Price_Lag1", "Spread_Brent_WTI"]
    bien_phu_thuoc = "WTI_Price"
    df_corr = phan_tich_tuong_quan_pearson(df_encoded, cac_bien_doc_lap, bien_phu_thuoc)

    # 4. Trực quan hóa
    cac_bien_ve = [bien_phu_thuoc] + cac_bien_doc_lap
    truc_quan_hoa_tuong_quan(df_encoded, cac_bien_ve, fig03_png, fig03_html, fig04_png, fig04_html)

    # Lưu tập dữ liệu đã biến đổi
    df_encoded.to_csv(file_transformed, index=False, encoding="utf-8")

    assert os.path.exists(file_transformed) and os.path.getsize(file_transformed) > 0, "Lỗi file transformed rỗng!"
    assert os.path.exists(fig03_html) and os.path.getsize(fig03_html) > 0, "Lỗi file fig03 HTML!"
    assert os.path.exists(fig04_html) and os.path.getsize(fig04_html) > 0, "Lỗi file fig04 HTML!"

    print(f"\n[THÀNH CÔNG] Đã lưu file transformed: {file_transformed} ({len(df_encoded)} dòng).")
    print("SELF-TEST BƯỚC 3: PASS 100%!")

if __name__ == "__main__":
    main()
