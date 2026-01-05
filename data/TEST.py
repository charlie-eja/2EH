import pandas as pd
import numpy as np

# 讀取 Excel（如果是 CSV 就用 pd.read_csv）
df = pd.read_excel("Isomer_Column_Process.xlsx",sheet_name='Sheet2')   # <- 改成你的檔案
print(df.columns.tolist())
# df = pd.read_csv("input.csv")

# 欄位名稱（照你貼的）
time_col = "Time"
value_cols = ["ML2EH_S232-IBAL", "ML2EH_S232-NBAL"]

# 1) 把錯誤訊息字串 -> NaN，並把數值欄轉成 numeric
for c in value_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")  # 不能轉的字串會變 NaN

# 2) Time 轉 datetime 並排序
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col)

# 3) 如果你是「間隔採樣」可能 Time 不一定每小時都有
#    先把它補成每小時一筆（依你的例子是每小時）
df = df.set_index(time_col).asfreq("H")  # 沒有的時間點會出現 NaN

# 4A) 線性插值（適合連續型數據）
df_interp = df.copy()
df_interp[value_cols] = df_interp[value_cols].interpolate(method="time")

# 4B) 如果你希望頭尾也不要 NaN（例如前面一整段都缺）
#     可用前值/後值補齊
df_interp[value_cols] = df_interp[value_cols].ffill().bfill()

# 5) 存回 Excel
df_interp = df_interp.reset_index()
df_interp.to_excel("output_filled.xlsx", index=False)

print("done -> output_filled.xlsx")
