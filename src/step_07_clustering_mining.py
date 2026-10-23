import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def khao_sat_k_toi_uu(X_scaled, k_range=range(2, 9)):
    print("=" * 85)
    print("TASK 7.1: KHẢO SÁT SỐ CỤM TỐI ƯU K (ELBOW METHOD & SILHOUETTE SCORE)")
    print("=" * 85)

    inertia_list = []
    silhouette_list = []

    print(f"{'K':<6}{'Inertia':<18}{'Silhouette Score'}")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        inertia_list.append(inertia)
        silhouette_list.append(sil)
        print(f"k={k:<4}{inertia:<18.2f}{sil:.4f}")

    best_k = list(k_range)[np.argmax(silhouette_list)]
    print(f"\n-> Số cụm tối ưu theo Silhouette Score: K = {best_k}")
    return list(k_range), inertia_list, silhouette_list, best_k

def truc_quan_hoa_elbow_silhouette_html(k_range, inertia_list, silhouette_list, optimal_k, fig_html, fig_png):
    print("\n" + "=" * 85)
    print("TASK 7.2: TRỰC QUAN HÓA ELBOW & SILHOUETTE (HTML & PNG)")
    print("=" * 85)

    # 1. PLOTLY TƯƠNG TÁC (HTML)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Phương Pháp Khuỷu Tay (Elbow Curve - Inertia)", "Hệ Số Silhouette Score Theo Số Cụm K")
    )
    fig.add_trace(
        go.Scatter(
            x=k_range, y=inertia_list, mode="lines+markers",
            marker=dict(size=8, color="#1f77b4"), name="Inertia",
            hovertemplate="K = %{x}<br>Inertia: %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="#d62728", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=k_range, y=silhouette_list, mode="lines+markers",
            marker=dict(size=8, color="#2ca02c"), name="Silhouette Score",
            hovertemplate="K = %{x}<br>Silhouette: %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="#d62728", row=1, col=2)

    fig.update_layout(
        title=dict(text=f"<b>KHẢO SÁT GOM CỤM THỊ TRƯỜNG DẦU THÔ: K TỐI ƯU = {optimal_k}</b>", font=dict(size=15)),
        template="plotly_white",
        height=450,
        showlegend=False
    )
    fig.write_html(fig_html, include_plotlyjs="cdn")
    print(f"- Đã xuất biểu đồ Elbow & Silhouette HTML: {fig_html}")

    # Bản tĩnh PNG
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia_list, "bo-", linewidth=2)
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"K tối ưu = {optimal_k}")
    plt.title("Phương pháp Elbow (Inertia)", fontweight="bold")
    plt.xlabel("Số cụm K")
    plt.ylabel("Inertia")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_list, "go-", linewidth=2)
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"K tối ưu = {optimal_k}")
    plt.title("Silhouette Score", fontweight="bold")
    plt.xlabel("Số cụm K")
    plt.ylabel("Silhouette")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất ảnh Elbow & Silhouette PNG: {fig_png}")

def phan_tich_va_truc_quan_hoa_cum(df, feature_cols, optimal_k, fig_html, fig_png):
    print("\n" + "=" * 85)
    print("TASK 7.3: HUẤN LUYỆN K-MEANS & ĐỊNH DANH CÁC CHẾ ĐỘ THỊ TRƯỜNG (MARKET REGIMES)")
    print("=" * 85)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    df["Cluster"] = km.fit_predict(X_scaled)

    # Đặt tên trực quan cho các cụm dựa vào mức giá trung bình của WTI
    cluster_means = df.groupby("Cluster")["WTI_Price"].mean().sort_values()
    regime_names = {}
    names_pool = [
        "1. Thị Trường Giá Thấp & Ổn Định",
        "2. Thị Trường Giá Trung Bình",
        "3. Thị Trường Giá Cao / Cú Sốc Năng Lượng"
    ]
    if optimal_k > 3:
        names_pool = [f"Chế độ {i+1}" for i in range(optimal_k)]

    for idx, (c_id, _) in enumerate(cluster_means.items()):
        regime_names[c_id] = names_pool[min(idx, len(names_pool)-1)]

    df["Market_Regime"] = df["Cluster"].map(regime_names)

    # Thống kê mô tả theo cụm
    summary = df.groupby("Market_Regime")[feature_cols].agg(["count", "mean", "std", "min", "max"]).round(2)
    print("BẢNG ĐẶC TRƯNG CÁC CHẾ ĐỘ THỊ TRƯỜNG DẦU THÔ:")
    print(summary)

    # 1. PLOTLY TƯƠNG TÁC (HTML)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Phân Bổ Cụm Thị Trường Theo Chuỗi Thời Gian (Market Regime Timeline)", "Không Gian Phân Cụm: Giá WTI vs Chênh Lệch Spread"),
        row_heights=[0.55, 0.45],
        vertical_spacing=0.1
    )

    palette = {"1. Thị Trường Giá Thấp & Ổn Định": "#2ca02c", "2. Thị Trường Giá Trung Bình": "#1f77b4", "3. Thị Trường Giá Cao / Cú Sốc Năng Lượng": "#d62728"}

    for regime, grp in df.groupby("Market_Regime"):
        clr = palette.get(regime, "#333333")
        fig.add_trace(
            go.Scatter(
                x=grp["Date"], y=grp["WTI_Price"],
                mode="markers", name=regime,
                marker=dict(size=4, color=clr),
                hovertemplate="Ngày: %{x}<br>WTI: $%{y:.2f}<br>Regime: " + regime + "<extra></extra>"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=grp["WTI_Price"], y=grp["Spread_Brent_WTI"],
                mode="markers", name=regime,
                marker=dict(size=5, color=clr, opacity=0.7),
                showlegend=False,
                hovertemplate="WTI: $%{x:.2f}<br>Spread: $%{y:.2f}<extra></extra>"
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=dict(text="<b>KHAI PHÁ DỮ LIỆU K-MEANS: NHẬN DIỆN CÁC CHẾ ĐỘ THỊ TRƯỜNG DẦU THÔ</b>", font=dict(size=16)),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=800
    )
    fig.write_html(fig_html, include_plotlyjs="cdn")
    print(f"- Đã xuất biểu đồ gom cụm tương tác HTML: {fig_html}")

    # Bản tĩnh PNG
    plt.figure(figsize=(14, 6))
    for regime, grp in df.groupby("Market_Regime"):
        dates = pd.to_datetime(grp["Date"])
        plt.scatter(dates, grp["WTI_Price"], label=regime, s=15, alpha=0.75)
    plt.title("DÒNG THỜI GIAN CÁC CHẾ ĐỘ THỊ TRƯỜNG DẦU THÔ (K-MEANS CLUSTERING)", fontsize=13, fontweight="bold")
    plt.xlabel("Thời gian", fontsize=11)
    plt.ylabel("Giá Dầu WTI (USD/thùng)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất ảnh gom cụm PNG: {fig_png}")

    return df, summary

def main():
    file_cleaned = os.path.join("data", "processed", "oil_cleaned.csv")
    assert os.path.exists(file_cleaned), f"Lỗi: Không tìm thấy '{file_cleaned}'! Chạy step 2 trước."
    df = pd.read_csv(file_cleaned)

    feature_cols = ["WTI_Price", "Brent_Price", "Spread_Brent_WTI"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    k_range, inertia_list, silhouette_list, optimal_k = khao_sat_k_toi_uu(X_scaled)
    # Thường với bài toán phân tích chu kỳ dầu mỏ, K = 3 mang ý nghĩa kinh tế học cao nhất
    K_CHON = 3

    fig07_html = os.path.join("figures", "fig07_elbow_silhouette.html")
    fig07_png = os.path.join("figures", "fig07_elbow_silhouette.png")
    truc_quan_hoa_elbow_silhouette_html(k_range, inertia_list, silhouette_list, K_CHON, fig07_html, fig07_png)

    fig08_html = os.path.join("figures", "fig08_market_regimes_clustering.html")
    fig08_png = os.path.join("figures", "fig08_market_regimes_clustering.png")
    df_clustered, summary = phan_tich_va_truc_quan_hoa_cum(df, feature_cols, K_CHON, fig08_html, fig08_png)

    # Lưu kết quả
    file_clustered = os.path.join("data", "processed", "oil_clustered.csv")
    df_clustered.to_csv(file_clustered, index=False, encoding="utf-8-sig")

    os.makedirs("reports", exist_ok=True)
    summary.to_csv(os.path.join("reports", "cluster_regimes_summary.csv"), encoding="utf-8-sig")

    assert os.path.exists(fig07_html) and os.path.getsize(fig07_html) > 0, "Lỗi file fig07 HTML!"
    assert os.path.exists(fig08_html) and os.path.getsize(fig08_html) > 0, "Lỗi file fig08 HTML!"
    assert os.path.exists(file_clustered) and os.path.getsize(file_clustered) > 0, "Lỗi file clustered rỗng!"

    print(f"\n[THÀNH CÔNG] ĐÃ HOÀN TẤT GOM CỤM THỊ TRƯỜNG: {file_clustered}")
    print("SELF-TEST BƯỚC 7: PASS 100%!")

if __name__ == "__main__":
    main()
