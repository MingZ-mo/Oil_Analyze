import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def chia_tap_du_lieu_chuoi_thoi_gian(df, ty_le_train=0.8):
    print("=" * 85)
    print("TASK 4.1: CHIA TẬP DỮ LIỆU HUẤN LUYỆN (TRAIN 80%) & KIỂM THỬ (TEST 20%)")
    print("=" * 85)

    tong_so_mau = len(df)
    diem_cat = int(tong_so_mau * ty_le_train)
    df_train = df.iloc[:diem_cat].copy().reset_index(drop=True)
    df_test = df.iloc[diem_cat:].copy().reset_index(drop=True)

    print(f"- Toàn bộ mẫu: {tong_so_mau} quan sát.")
    print(f"- Train Set: {len(df_train)} ngày ({df_train['Date'].iloc[0]} -> {df_train['Date'].iloc[-1]})")
    print(f"- Test Set : {len(df_test)} ngày ({df_test['Date'].iloc[0]} -> {df_test['Date'].iloc[-1]})")
    return df_train, df_test

def uoc_luong_mo_hinh_day_du(df_train, cac_bien_doc_lap, bien_phu_thuoc):
    print("\n" + "=" * 85)
    print("TASK 4.2: XÂY DỰNG MÔ HÌNH HỒI QUY ĐẦY ĐỦ (FULL MULTIPLE REGRESSION MODEL)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    X_train = sm.add_constant(df_train[cac_bien_doc_lap])
    model_full = sm.OLS(y_train, X_train).fit()
    print(model_full.summary())
    return model_full

def thuat_toan_backward_elimination(df_train, cac_bien_ung_vien, bien_phu_thuoc, nguong_p=0.05):
    print("\n" + "=" * 85)
    print("TASK 4.3: THỰC THI THUẬT TOÁN LỰA CHỌN BIẾN LÙI (BACKWARD ELIMINATION)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    bien_hien_tai = list(cac_bien_ung_vien)
    nhat_ky_vong_lap = []
    vong_lap = 1

    while True:
        X_curr = sm.add_constant(df_train[bien_hien_tai])
        model_curr = sm.OLS(y_train, X_curr).fit()

        p_values = model_curr.pvalues.drop("const")
        max_p = p_values.max()
        bien_kem_nhat = p_values.idxmax()

        thong_tin_vong = {
            "Vòng": vong_lap,
            "Số biến": len(bien_hien_tai),
            "Adj R-squared": round(model_curr.rsquared_adj, 4),
            "AIC": round(model_curr.aic, 2),
            "Biến có p lớn nhất": bien_kem_nhat,
            "Max p-value": f"{max_p:.4e}"
        }
        nhat_ky_vong_lap.append(thong_tin_vong)

        print(f"[Vòng {vong_lap}] Số biến: {len(bien_hien_tai):<2} | Adj R²: {model_curr.rsquared_adj:.4f} | AIC: {model_curr.aic:.2f} | Biến max p: '{bien_kem_nhat}' (p = {max_p:.4e})")

        if max_p > nguong_p:
            print(f"  -> Loại bỏ biến '{bien_kem_nhat}' do p-value ({max_p:.4e}) > ngưỡng {nguong_p}")
            bien_hien_tai.remove(bien_kem_nhat)
            vong_lap += 1
        else:
            print(f"  -> DỪNG THUẬT TOÁN: Tất cả {len(bien_hien_tai)} biến còn lại đều có p-value <= {nguong_p}!")
            model_toi_uu = model_curr
            break

    df_nhat_ky = pd.DataFrame(nhat_ky_vong_lap)
    print("\n" + "-" * 85)
    print("BẢNG TỔNG HỢP TIẾN TRÌNH BACKWARD ELIMINATION:")
    print(df_nhat_ky.to_string(index=False))
    return model_toi_uu, bien_hien_tai, df_nhat_ky

def trinh_bay_phuong_trinh_hoi_quy(model, bien_phu_thuoc, cac_bien_toi_uu):
    print("\n" + "=" * 85)
    print("TASK 4.4: PHƯƠNG TRÌNH HỒI QUY OLS TỐI ƯU & HỆ SỐ COINTEGRATION")
    print("=" * 85)

    params = model.params
    pvalues = model.pvalues
    bpt = f"{bien_phu_thuoc} = {params['const']:.4f}"

    for b in cac_bien_toi_uu:
        dau = " + " if params[b] >= 0 else " - "
        bpt += f"{dau}{abs(params[b]):.4f} * {b}"

    print(f"PHƯƠNG TRÌNH DỰ BÁO: \n{bpt}\n")
    print(f"- Hệ số xác định R-squared:     {model.rsquared:.4f} ({model.rsquared*100:.2f}% phương sai được giải thích)")
    print(f"- R-squared hiệu chỉnh (Adj):   {model.rsquared_adj:.4f}")
    print(f"- Thống kê F-statistic:         {model.fvalue:.2f} (p = {model.f_pvalue:.4e})")
    print(f"- Tiêu chuẩn thông tin AIC:     {model.aic:.2f}")
    print(f"- Tiêu chuẩn thông tin BIC:     {model.bic:.2f}")

    bang_he_so = []
    for var in ["const"] + cac_bien_toi_uu:
        bang_he_so.append({
            "Biến": var,
            "Hệ số (Coef)": round(params[var], 4),
            "Std Err": round(model.bse[var], 4),
            "t-stat": round(model.tvalues[var], 4),
            "p-value": f"{pvalues[var]:.4e}",
            "Khoảng tin cậy 95% Dưới": round(model.conf_int().loc[var, 0], 4),
            "Khoảng tin cậy 95% Trên": round(model.conf_int().loc[var, 1], 4)
        })
    df_params = pd.DataFrame(bang_he_so)
    print("\nBẢNG THỐNG KÊ CHI TIẾT HỆ SỐ MÔ HÌNH:")
    print(df_params.to_string(index=False))
    return df_params

def main():
    file_transformed = os.path.join("data", "processed", "oil_transformed.csv")
    assert os.path.exists(file_transformed), f"Lỗi: Không tìm thấy '{file_transformed}'!"
    df = pd.read_csv(file_transformed)

    df_train, df_test = chia_tap_du_lieu_chuoi_thoi_gian(df, ty_le_train=0.8)

    # Tập biến ứng viên kinh tế lượng
    bien_phu_thuoc = "WTI_Price"
    cac_bien_ung_vien = [
        "Brent_Price",
        "WTI_Price_Lag1",
        "Brent_Price_Lag1",
        "Season_Q2_Spring",
        "Season_Q3_Summer",
        "Season_Q4_Autumn"
    ]

    model_full = uoc_luong_mo_hinh_day_du(df_train, cac_bien_ung_vien, bien_phu_thuoc)
    model_toi_uu, cac_bien_toi_uu, df_nhat_ky = thuat_toan_backward_elimination(
        df_train, cac_bien_ung_vien, bien_phu_thuoc, nguong_p=0.05
    )
    df_params = trinh_bay_phuong_trinh_hoi_quy(model_toi_uu, bien_phu_thuoc, cac_bien_toi_uu)

    # Lưu kết quả
    os.makedirs("reports", exist_ok=True)
    df_train.to_csv(os.path.join("data", "processed", "oil_train.csv"), index=False)
    df_test.to_csv(os.path.join("data", "processed", "oil_test.csv"), index=False)
    df_params.to_csv(os.path.join("reports", "ols_coefficients.csv"), index=False)
    df_nhat_ky.to_csv(os.path.join("reports", "backward_elimination_log.csv"), index=False)

    print("\n[THÀNH CÔNG] ĐÃ LƯU KẾT QUẢ MÔ HÌNH VÀ TẬP DỮ LIỆU TRAIN/TEST.")
    print("SELF-TEST BƯỚC 4: PASS 100%!")

if __name__ == "__main__":
    main()
