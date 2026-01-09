import os
import pandas as pd

def com_file(input_file,input_file_weather,output_file,):
    # ====== 檔案路徑 ======
    df_main = pd.read_excel(input_file, sheet_name='Sheet2')

    df_w = pd.read_csv(
        input_file_weather,
        encoding="utf-8-sig"
    )

    # ====== 清理氣象欄名（去 BOM / 空白）======
    df_w.columns = (
        df_w.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # ====== 時間欄位 ======
    MAIN_TIME_COL = "Time"
    WX_TIME_COL = "觀測時間(hour)"

    df_main[MAIN_TIME_COL] = pd.to_datetime(df_main[MAIN_TIME_COL], errors="coerce")
    df_w[WX_TIME_COL] = pd.to_datetime(df_w[WX_TIME_COL], errors="coerce")

    # ====== 主表數值清理（千分位）======
    for c in df_main.columns:
        if c == MAIN_TIME_COL:
            continue
        if df_main[c].dtype == "object":
            df_main[c] = pd.to_numeric(
                df_main[c].astype(str).str.replace(",", "", regex=False),
                errors="ignore"
            )

    # ====== 只保留需要的氣象欄位 + 改英文名 ======
    weather_cols_map = {
        "測站氣壓(hPa)": "Station Pressure",
        "海平面氣壓(hPa)": "Sea Level Pressure",
        "氣溫(℃)": "Air Temperature",
        "露點溫度(℃)": "Dew Point Temperature",
        "相對溼度(%)": "Relative Humidity",
        "風速(m/s)": "Wind Speed",
        "風向(360degree)": "Wind Direction",
        "最大陣風(m/s)": "Maximum Gust",
        "最大陣風風向(360degree)": "Maximum Gust Direction",
        "降水量(mm)": "Precipitation",
    }

    # 確認欄位存在
    missing = [c for c in weather_cols_map if c not in df_w.columns]
    if missing:
        raise KeyError(f"氣象CSV缺少欄位: {missing}")

    df_w_sel = df_w[[WX_TIME_COL] + list(weather_cols_map.keys())].rename(
        columns=weather_cols_map
    )

    # ====== 去重（同一小時只留一筆）======
    df_w_sel = df_w_sel.dropna(subset=[WX_TIME_COL]) \
                       .drop_duplicates(subset=[WX_TIME_COL], keep="first")

    # ====== 合併（left join）======
    df_out = df_main.merge(
        df_w_sel,
        how="left",
        left_on=MAIN_TIME_COL,
        right_on=WX_TIME_COL
    ).drop(columns=[WX_TIME_COL])

    # ====== 輸出 ======
    os.makedirs("data", exist_ok=True)
    df_out.to_excel(output_file, index=False)

    print(f"✔ 完成：{output_file}")
