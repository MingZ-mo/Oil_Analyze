import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
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

def tinh_toan_cac_chi_so_goc_da_sgu(y_true, y_pred, n_obs, k_features, llf):
    ess = np.sum((y_true - y_pred) ** 2)
    y_bar = np.mean(y_true)
    tss = np.sum((y_true - y_bar) ** 2)
    r2 = 1.0 - (ess / tss)
    adj_r2 = 1.0 - ((ess / (n_obs - k_features - 1)) / (tss / (n_obs - 1)))
    aic = -2.0 * llf + 2.0 * (k_features + 1)
    bic = -2.0 * llf + np.log(n_obs) * (k_features + 1)
    return {"ESS": ess, "TSS": tss, "R2": r2, "Adj_R2": adj_r2, "AIC": aic, "BIC": bic}

def huan_luyen_va_so_sanh_mo_hinh(df_train, df_test, bien_phu_thuoc, features):
    print("=" * 85)
    print("TASK 6.1: TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH THEO CÔNG THỨC GỐC DA-SGU")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    y_test = df_test[bien_phu_thuoc]

    # 1. OLS Tuyến tính chuẩn
    X_train_lin = sm.add_constant(df_train[features])
    X_test_lin = sm.add_constant(df_test[features])
    m_lin = sm.OLS(y_train, X_train_lin).fit()

    # 2. OLS Robust HC3 (Chống Heteroscedasticity)
    m_hc3 = sm.OLS(y_train, X_train_lin).fit(cov_type="HC3")

    calc_lin = tinh_toan_cac_chi_so_goc_da_sgu(
        y_train.values, m_lin.fittedvalues.values, len(y_train), len(features), m_lin.llf
    )
    print(f"- ESS: {calc_lin['ESS']:,.2f} | TSS: {calc_lin['TSS']:,.2f}")
    print(f"- R2 : {calc_lin['R2']:.4f} | Adj R2: {calc_lin['Adj_R2']:.4f} | AIC: {calc_lin['AIC']:.2f} | BIC: {calc_lin['BIC']:.2f}")

    pred_test_lin = m_lin.predict(X_test_lin)
    rmse_lin = np.sqrt(mean_squared_error(y_test, pred_test_lin))
    mae_lin = mean_absolute_error(y_test, pred_test_lin)
    mape_lin = mean_absolute_percentage_error(y_test, pred_test_lin) * 100.0

    # 3. Mô hình Log-Log OLS
    train_log = df_train.copy()
    test_log = df_test.copy()
    train_log["ln_Y"] = np.log(train_log[bien_phu_thuoc])
    test_log["ln_Y"] = np.log(test_log[bien_phu_thuoc])

    log_features = []
    for col in features:
        if col.startswith("Season_"):
            log_features.append(col)
        else:
            train_log["ln_" + col] = np.log(train_log[col])
            test_log["ln_" + col] = np.log(test_log[col])
            log_features.append("ln_" + col)

    X_train_log = sm.add_constant(train_log[log_features])
    X_test_log = sm.add_constant(test_log[log_features])
    m_log = sm.OLS(train_log["ln_Y"], X_train_log).fit()

    pred_test_log_ln = m_log.predict(X_test_log)
    pred_test_log = np.exp(pred_test_log_ln)

    rmse_log = np.sqrt(mean_squared_error(y_test, pred_test_log))
    mae_log = mean_absolute_error(y_test, pred_test_log)
    mape_log = mean_absolute_percentage_error(y_test, pred_test_log) * 100.0

    bang_so_sanh = [
        {
            "Mô hình": "1. Linear OLS (Chuẩn)",
            "Train R2": round(m_lin.rsquared, 4),
            "Train Adj R2": round(m_lin.rsquared_adj, 4),
            "AIC": round(m_lin.aic, 2),
            "BIC": round(m_lin.bic, 2),
            "Test RMSE ($)": round(rmse_lin, 4),
            "Test MAE ($)": round(mae_lin, 4),
            "Test MAPE (%)": round(mape_lin, 2)
        },
        {
            "Mô hình": "2. Robust OLS (HC3)",
            "Train R2": round(m_hc3.rsquared, 4),
            "Train Adj R2": round(m_hc3.rsquared_adj, 4),
            "AIC": round(m_lin.aic, 2),
            "BIC": round(m_lin.bic, 2),
            "Test RMSE ($)": round(rmse_lin, 4),
            "Test MAE ($)": round(mae_lin, 4),
            "Test MAPE (%)": round(mape_lin, 2)
        },
        {
            "Mô hình": "3. Log-Log OLS",
            "Train R2": round(m_log.rsquared, 4),
            "Train Adj R2": round(m_log.rsquared_adj, 4),
            "AIC": round(m_log.aic, 2),
            "BIC": round(m_log.bic, 2),
            "Test RMSE ($)": round(rmse_log, 4),
            "Test MAE ($)": round(mae_log, 4),
            "Test MAPE (%)": round(mape_log, 2)
        }
    ]

    df_so_sanh = pd.DataFrame(bang_so_sanh)
    print("\n" + "=" * 85)
    print("BẢNG SO SÁNH HIỆU NĂNG CÁC KIẾN TRÚC MÔ HÌNH DỰ BÁO GIÁ DẦU:")
    print("=" * 85)
    print(df_so_sanh.to_string(index=False))

    return m_lin, pred_test_lin, df_so_sanh

def truc_quan_hoa_du_bao_html(df_train, df_test, pred_test, fig_html, fig_png):
    print("\n" + "=" * 85)
    print("TASK 6.2: TRỰC QUAN HÓA DỰ BÁO TRÊN TẬP TEST (HTML & PNG)")
    print("=" * 85)

    # 1. BIỂU ĐỒ PLOTLY TƯƠNG TÁC (HTML)
    fig_eval = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Kiểm Thử Dự Báo Giá Dầu WTI: Thực Tế vs Dự Báo (Test Set)", "Sai Số Tuyệt Đối Hàng Ngày (|Error| = |Actual - Forecast|) ($)"),
        row_heights=[0.7, 0.3]
    )

    # Actual Test
    fig_eval.add_trace(
        go.Scatter(
            x=df_test["Date"], y=df_test["WTI_Price"],
            mode="lines+markers", name="WTI Thực Tế (Test Set)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            hovertemplate="Ngày: %{x}<br>Thực tế: $%{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Predicted Test
    fig_eval.add_trace(
        go.Scatter(
            x=df_test["Date"], y=pred_test,
            mode="lines", name="WTI Dự Báo (OLS Model)",
            line=dict(color="#d62728", width=2, dash="dash"),
            hovertemplate="Ngày: %{x}<br>Dự báo: $%{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Error
    errors = np.abs(df_test["WTI_Price"].values - pred_test.values)
    fig_eval.add_trace(
        go.Bar(
            x=df_test["Date"], y=errors,
            name="Sai Số Tuyệt Đối ($)",
            marker_color="#ff7f0e", opacity=0.7,
            hovertemplate="Ngày: %{x}<br>Sai số: $%{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    fig_eval.update_layout(
        title=dict(text="<b>ĐÁNH GIÁ DỰ BÁO GIÁ DẦU WTI TRÊN TẬP KIỂM THỬ ĐỘC LẬP (OUT-OF-SAMPLE TEST)</b>", font=dict(size=16)),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 Tháng", step="month", stepmode="backward"),
                    dict(count=3, label="3 Tháng", step="month", stepmode="backward"),
                    dict(count=6, label="6 Tháng", step="month", stepmode="backward"),
                    dict(step="all", label="Toàn Bộ Test")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.08),
            type="date"
        ),
        height=750
    )
    fig_eval.write_html(fig_html, include_plotlyjs="cdn")
    print(f"- Đã xuất biểu đồ dự báo tương tác HTML: {fig_html}")

    # Bản tĩnh PNG
    plt.figure(figsize=(14, 6))
    dates_test = pd.to_datetime(df_test["Date"])
    plt.plot(dates_test, df_test["WTI_Price"], label="WTI Thực tế (Test Set)", color="#1f77b4", linewidth=2)
    plt.plot(dates_test, pred_test, label="WTI Dự báo OLS", color="#d62728", linestyle="--", linewidth=2)
    plt.fill_between(dates_test, pred_test - 1.5, pred_test + 1.5, color="#d62728", alpha=0.15, label="Dải tin cậy ±$1.50")
    plt.title("ĐÁNH GIÁ ĐỘ CHÍNH XÁC DỰ BÁO TRÊN TẬP KIỂM THỬ (261 NGÀY GIAO DỊCH)", fontsize=13, fontweight="bold")
    plt.xlabel("Thời gian", fontsize=11)
    plt.ylabel("Giá Dầu WTI (USD/thùng)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã xuất ảnh dự báo PNG: {fig_png}")

def main():
    file_train = os.path.join("data", "processed", "oil_train.csv")
    file_test = os.path.join("data", "processed", "oil_test.csv")
    assert os.path.exists(file_train) and os.path.exists(file_test), "Lỗi: Hãy chạy step 4 trước!"

    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    features = ["Brent_Price", "WTI_Price_Lag1", "Brent_Price_Lag1", "Season_Q2_Spring"]
    bien_phu_thuoc = "WTI_Price"

    m_lin, pred_test, df_so_sanh = huan_luyen_va_so_sanh_mo_hinh(df_train, df_test, bien_phu_thuoc, features)

    fig_html = os.path.join("figures", "fig06_forecast_evaluation.html")
    fig_png = os.path.join("figures", "fig06_forecast_evaluation.png")
    truc_quan_hoa_du_bao_html(df_train, df_test, pred_test, fig_html, fig_png)

    # Lưu kết quả
    os.makedirs("reports", exist_ok=True)
    df_so_sanh.to_csv(os.path.join("reports", "model_comparison.csv"), index=False, encoding="utf-8-sig")

    df_test_out = df_test[["Date", "WTI_Price", "Brent_Price"]].copy()
    df_test_out["WTI_Predicted_OLS"] = np.round(pred_test.values, 4)
    df_test_out["Absolute_Error"] = np.round(np.abs(df_test_out["WTI_Price"] - df_test_out["WTI_Predicted_OLS"]), 4)
    df_test_out.to_csv(os.path.join("reports", "test_predictions.csv"), index=False, encoding="utf-8-sig")

    assert os.path.exists(fig_html) and os.path.getsize(fig_html) > 0, "Lỗi file fig06 HTML!"
    print(f"\n[THÀNH CÔNG] ĐÃ HOÀN TẤT ĐÁNH GIÁ VÀ XUẤT BÁO CÁO DỰ BÁO: {fig_html}")
    print("SELF-TEST BƯỚC 6: PASS 100%!")

if __name__ == "__main__":
    main()
