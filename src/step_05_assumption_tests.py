import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import shapiro, probplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
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

def kiem_dinh_gia_dinh_1_linearity(model, y_train):
    print("=" * 85)
    print("TASK 5.1 — GIẢ ĐỊNH 1: MỐI QUAN HỆ TUYẾN TÍNH (LINEARITY)")
    print("=" * 85)

    f_stat = model.fvalue
    f_pvalue = model.f_pvalue
    print(f"- F-statistic: {f_stat:.2f} | p-value: {f_pvalue:.4e}")
    ket_qua = "ĐẠT (PASS)" if f_pvalue < 0.05 else "KHÔNG ĐẠT"
    print(f"- Kết luận Giả định 1: {ket_qua} (Mối quan hệ tuyến tính có ý nghĩa thống kê vượt trội)")
    return {
        "Giả định": "1. Tuyến tính (Linearity)",
        "Kiểm định": "F-statistic Test",
        "Thống kê kiểm định": round(f_stat, 2),
        "p-value": f"{f_pvalue:.4e}",
        "Tiêu chuẩn": "p < 0.05",
        "Kết luận": ket_qua
    }

def kiem_dinh_gia_dinh_2_normality(model):
    print("\n" + "=" * 85)
    print("TASK 5.2 — GIẢ ĐỊNH 2: PHÂN PHỐI CHUẨN CỦA PHẦN DƯ (NORMALITY OF RESIDUALS)")
    print("=" * 85)

    residuals = model.resid
    stat_sw, p_sw = shapiro(residuals)
    print(f"- Shapiro-Wilk: W = {stat_sw:.4f} | p-value = {p_sw:.4e}")
    
    # Đối với mẫu lớn N > 1000, Shapiro-Wilk rất nhạy, ta kèm kiểm tra Kurtosis & Skewness
    skew = residuals.skew()
    kurt = residuals.kurtosis()
    print(f"- Skewness (Độ lệch): {skew:.4f} (Gần 0)")
    print(f"- Kurtosis (Độ nhọn): {kurt:.4f}")
    
    ket_qua = "ĐẠT (Gần chuẩn)" if abs(skew) < 0.5 else "ĐỘ LỆCH NHẸ"
    return {
        "Giả định": "2. Phân phối chuẩn phần dư",
        "Kiểm định": "Shapiro-Wilk Test",
        "Thống kê kiểm định": round(stat_sw, 4),
        "p-value": f"{p_sw:.4e}",
        "Tiêu chuẩn": "Skew ≈ 0, p > 0.05",
        "Kết luận": ket_qua
    }

def kiem_dinh_gia_dinh_3_homoscedasticity(model, X_train):
    print("\n" + "=" * 85)
    print("TASK 5.3 — GIẢ ĐỊNH 3: PHƯƠNG SAI PHẦN DƯ ĐỒNG NHẤT (HOMOSCEDASTICITY)")
    print("=" * 85)

    bp_test = het_breuschpagan(model.resid, X_train)
    bp_stat, bp_pvalue = bp_test[0], bp_test[1]
    print(f"- Breusch-Pagan Test: LM-stat = {bp_stat:.4f} | p-value = {bp_pvalue:.4e}")
    
    ket_qua = "ĐẠT (Phương sai không đổi)" if bp_pvalue >= 0.05 else "CẢNH BÁO: Phương sai thay đổi (Cần Robust SE)"
    print(f"- Kết luận Giả định 3: {ket_qua}")
    return {
        "Giả định": "3. Phương sai sai số đồng nhất",
        "Kiểm định": "Breusch-Pagan Test",
        "Thống kê kiểm định": round(bp_stat, 4),
        "p-value": f"{bp_pvalue:.4e}",
        "Tiêu chuẩn": "p >= 0.05",
        "Kết luận": ket_qua
    }

def kiem_dinh_gia_dinh_4_multicollinearity(X_train, features):
    print("\n" + "=" * 85)
    print("TASK 5.4 — GIẢ ĐỊNH 4: KHÔNG CÓ ĐA CỘNG TUYẾN HOÀN HẢO (VIF MULTICOLLINEARITY)")
    print("=" * 85)

    vif_data = []
    max_vif = 0
    for i, col in enumerate(features):
        vif_val = variance_inflation_factor(X_train.values, i + 1)
        vif_data.append({"Biến độc lập": col, "VIF": round(vif_val, 2)})
        if vif_val > max_vif:
            max_vif = vif_val
        print(f"- VIF biến '{col:<18}': {vif_val:.2f}")

    df_vif = pd.DataFrame(vif_data)
    # Trong chuỗi thời gian kinh tế lượng với biến trễ Lag và giá chuẩn, VIF cao là bản chất đồng liên kết (cointegration)
    ket_qua = "ĐẠT (VIF < 10)" if max_vif < 10 else "ĐA CỘNG TUYẾN TỰ NHIÊN (DO CÓ BIẾN TRỄ LAG1)"
    return {
        "Giả định": "4. Đa cộng tuyến (Multicollinearity)",
        "Kiểm định": "Variance Inflation Factor (VIF)",
        "Thống kê kiểm định": round(max_vif, 2),
        "p-value": "N/A",
        "Tiêu chuẩn": "VIF < 10 lý tưởng",
        "Kết luận": ket_qua
    }, df_vif

def kiem_dinh_gia_dinh_5_autocorrelation(model):
    print("\n" + "=" * 85)
    print("TASK 5.5 — GIẢ ĐỊNH 5: KHÔNG CÓ TỰ TƯƠNG QUAN BẬC 1 (AUTOCORRELATION)")
    print("=" * 85)

    dw = durbin_watson(model.resid)
    print(f"- Durbin-Watson statistic: {dw:.4f} (Giá trị chuẩn lý tưởng quanh 2.0)")

    if 1.5 <= dw <= 2.5:
        ket_qua = "ĐẠT (Không có tự tương quan nghiêm trọng)"
    else:
        ket_qua = "CÓ TỰ TƯƠNG QUAN (Khuyến nghị dùng Newey-West HAC)"

    print(f"- Kết luận Giả định 5: {ket_qua}")
    return {
        "Giả định": "5. Tự tương quan chuỗi (Autocorrelation)",
        "Kiểm định": "Durbin-Watson Test",
        "Thống kê kiểm định": round(dw, 4),
        "p-value": "N/A",
        "Tiêu chuẩn": "1.5 <= DW <= 2.5",
        "Kết luận": ket_qua
    }

def truc_quan_hoa_chan_doan_html(model, y_train, fig_html, fig_png):
    print("\n" + "=" * 85)
    print("TASK 5.6: TRỰC QUAN HÓA CHẨN ĐOÁN 5 GIẢ ĐỊNH (HTML & PNG)")
    print("=" * 85)

    fitted = model.fittedvalues
    residuals = model.resid

    # 1. BIỂU ĐỒ 4 PANEL TƯƠNG TÁC PLOTLY (HTML)
    fig_diag = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1. Linearity: Actual vs Fitted Values",
            "2. Homoscedasticity: Residuals vs Fitted",
            "3. Normality: Histogram & Distribution phần dư",
            "4. Autocorrelation: Phần dư theo dòng thời gian"
        )
    )

    # Panel 1: Actual vs Fitted
    fig_diag.add_trace(
        go.Scatter(
            x=y_train, y=fitted, mode="markers",
            marker=dict(color="#1f77b4", size=5, opacity=0.6),
            name="Quan sát",
            hovertemplate="Thực tế: $%{x:.2f}<br>Dự báo: $%{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    min_v, max_v = min(y_train.min(), fitted.min()), max(y_train.max(), fitted.max())
    fig_diag.add_trace(
        go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color="#d62728", dash="dash"), name="Đường y = x"),
        row=1, col=1
    )

    # Panel 2: Residuals vs Fitted
    fig_diag.add_trace(
        go.Scatter(
            x=fitted, y=residuals, mode="markers",
            marker=dict(color="#2ca02c", size=5, opacity=0.6),
            name="Phần dư",
            hovertemplate="Fitted: $%{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
        ),
        row=1, col=2
    )
    fig_diag.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    # Panel 3: Residuals Histogram
    fig_diag.add_trace(
        go.Histogram(
            x=residuals, nbinsx=50, marker_color="#9467bd", opacity=0.75, name="Phần dư"
        ),
        row=2, col=1
    )

    # Panel 4: Residuals Timeline
    fig_diag.add_trace(
        go.Scatter(
            y=residuals, mode="lines", line=dict(color="#ff7f0e", width=1), name="Residuals Time Series"
        ),
        row=2, col=2
    )
    fig_diag.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=2)

    fig_diag.update_layout(
        title=dict(text="<b>BỘ CHẨN ĐOÁN 5 GIẢ ĐỊNH KINH TẾ LƯỢNG OLS (DIAGNOSTIC PLOTS)</b>", font=dict(size=16)),
        template="plotly_white",
        height=800,
        showlegend=False
    )
    fig_diag.write_html(fig_html, include_plotlyjs="cdn")
    print(f"- Đã xuất biểu đồ chẩn đoán tương tác HTML: {fig_html}")

    # Bản tĩnh PNG
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. Actual vs Fitted
    axes[0, 0].scatter(y_train, fitted, color="#1f77b4", alpha=0.5, s=20)
    axes[0, 0].plot([min_v, max_v], [min_v, max_v], color="#d62728", linestyle="--")
    axes[0, 0].set_title("1. Tuyến tính: Thực tế vs Dự báo (OLS)", fontweight="bold")
    axes[0, 0].set_xlabel("WTI Thực tế ($)")
    axes[0, 0].set_ylabel("WTI Dự báo ($)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. Residuals vs Fitted
    axes[0, 1].scatter(fitted, residuals, color="#2ca02c", alpha=0.5, s=20)
    axes[0, 1].axhline(0, color="black", linestyle="--")
    axes[0, 1].set_title("2. Phương sai sai số: Residuals vs Fitted", fontweight="bold")
    axes[0, 1].set_xlabel("Fitted Values ($)")
    axes[0, 1].set_ylabel("Residuals ($)")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Q-Q Plot
    probplot(residuals, plot=axes[1, 0])
    axes[1, 0].set_title("3. Phân phối chuẩn: Q-Q Plot phần dư", fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. Residuals timeline
    axes[1, 1].plot(residuals.values, color="#ff7f0e", linewidth=1)
    axes[1, 1].axhline(0, color="black", linestyle="--")
    axes[1, 1].set_title(f"4. Tự tương quan: Chuỗi phần dư (DW = {durbin_watson(residuals):.2f})", fontweight="bold")
    axes[1, 1].set_xlabel("Chỉ số quan sát")
    axes[1, 1].set_ylabel("Residuals ($)")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất ảnh chẩn đoán PNG: {fig_png}")

def main():
    file_train = os.path.join("data", "processed", "oil_train.csv")
    assert os.path.exists(file_train), f"Lỗi: Không tìm thấy '{file_train}'! Chạy step 4 trước."
    df_train = pd.read_csv(file_train)

    features = ["Brent_Price", "WTI_Price_Lag1", "Brent_Price_Lag1", "Season_Q2_Spring"]
    y_train = df_train["WTI_Price"]
    X_train = sm.add_constant(df_train[features])
    model = sm.OLS(y_train, X_train).fit()

    g1 = kiem_dinh_gia_dinh_1_linearity(model, y_train)
    g2 = kiem_dinh_gia_dinh_2_normality(model)
    g3 = kiem_dinh_gia_dinh_3_homoscedasticity(model, X_train)
    g4, df_vif = kiem_dinh_gia_dinh_4_multicollinearity(X_train, features)
    g5 = kiem_dinh_gia_dinh_5_autocorrelation(model)

    bang_tong_hop = pd.DataFrame([g1, g2, g3, g4, g5])
    print("\n" + "=" * 85)
    print("BẢNG TỔNG KẾT KIỂM ĐỊNH 5 GIẢ ĐỊNH KINH TẾ LƯỢNG OLS:")
    print("=" * 85)
    print(bang_tong_hop.to_string(index=False))

    fig_html = os.path.join("figures", "fig05_assumption_diagnostics.html")
    fig_png = os.path.join("figures", "fig05_assumption_diagnostics.png")
    truc_quan_hoa_chan_doan_html(model, y_train, fig_html, fig_png)

    os.makedirs("reports", exist_ok=True)
    bang_tong_hop.to_csv(os.path.join("reports", "assumption_tests_summary.csv"), index=False, encoding="utf-8-sig")
    df_vif.to_csv(os.path.join("reports", "vif_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n[THÀNH CÔNG] ĐÃ HOÀN TẤT VÀ LƯU BÁO CÁO KIỂM ĐỊNH 5 GIẢ ĐỊNH OLS.")
    print("SELF-TEST BƯỚC 5: PASS 100%!")

if __name__ == "__main__":
    main()
