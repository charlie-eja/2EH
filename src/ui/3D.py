import json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# ============ 儲坑維度 ============
ny, nx, nz = 5, 27, 30  # y, x, depth(z)

x_counts = [3, 3, 4, 4, 4, 5, 4]
doors    = [1, 2, 3, 3, 4, 5, 6]

# ============ 5 種廢棄物（示例；請換成你要的正式名稱/熱值/廠區）===========
# 類別 id = 0..4（mat3d 裡存這個）
wastes = [
    {"id": 0, "code": "D-0202", "name": "廢樹脂（D-0201除外）", "kcal": 9500, "site": "南廠"},
    {"id": 1, "code": "D-0201", "name": "廢離子交換樹脂",       "kcal": 7800, "site": "北廠"},
    {"id": 2, "code": "D-0299", "name": "廢塑膠混合物",         "kcal": 8800, "site": "中廠"},
    {"id": 3, "code": "D-0699", "name": "廢紙混合物",           "kcal": 3200, "site": "一廠"},
    {"id": 4, "code": "D-0901", "name": "有機性污泥",           "kcal": 1200, "site": "二廠"},
    {"id": 5, "code": "-", "name": "空",   "kcal": 0, "site": ""},
]
kcal_lookup = np.array([w["kcal"] for w in wastes], dtype=float)

segments = []
x0 = 0
for w, d in zip(x_counts, doors):
    segments.append({
        "door": d,
        "x0": x0,
        "x1": x0 + w
    })
    x0 += w

nx = x0  # 總欄數



# ============ 你的 3D 類別資料 mat3d[y,x,z]（0..4）===========
# 這裡用隨機示例；請換成你的資料
# mat3d = np.random.randint(0, len(wastes), size=(ny, nx, nz))

import numpy as np

# 1. 定義維度 (y, x, z)
ny, nx, nz = 5, 27, 30
shape = (ny, nx, nz)
mat3d = np.zeros(shape, dtype=int)

def filled_clustered_random(target_array, y_range, x_range, z_range, val_a, val_b):
    """
    在指定範圍內填入兩個數字，並透過簡單的平滑處理讓數字盡可能聚集。
    """
    y_s, y_e = y_range
    x_s, x_e = x_range
    z_s, z_e = z_range
    
    sub_shape = (y_e - y_s + 1,
                 x_e - x_s + 1,
                 z_e - z_s + 1)
    
    noise = np.random.rand(*sub_shape)
    
    from scipy.ndimage import uniform_filter
    smooth_noise = uniform_filter(noise, size=2)
    
    mask = smooth_noise > 0.5
    res = np.where(mask, val_a, val_b)
    
    target_array[y_s:y_e+1, x_s:x_e+1, z_s:z_e+1] = res


# 2. 依照條件填入數值（注意：現在是 [y, x, z]）

# 條件 1: x(0-5), y(0-4), z(0-12) -> 4
mat3d[0:5, 0:6, 0:13] = 4

# 條件 2: x(0-5), y(0-4), z(12-29) -> 5
mat3d[0:5, 0:6, 12:30] = 5

# 條件 3: x(6-17), y(0-4), z(0-15) -> 1 跟 3
filled_clustered_random(
    mat3d,
    y_range=(0, 4),
    x_range=(6, 17),
    z_range=(0, 15),
    val_a=1,
    val_b=3
)

# 條件 4: x(6-17), y(0-4), z(15-29) -> 5
mat3d[0:5, 6:18, 15:30] = 5

# 條件 5: x(18-26), y(0-4), z(0-18) -> 2 跟 0
filled_clustered_random(
    mat3d,
    y_range=(0, 4),
    x_range=(18, 26),
    z_range=(0, 18),
    val_a=2,
    val_b=0
)

# 條件 6: x(18-26), y(0-4), z(18-29) -> 5
mat3d[0:5, 18:27, 18:30] = 5

print("3D Array 生成完畢，形狀為:", mat3d.shape)


# ============ 3D：每類一個 trace（方便 checkbox 切換）===========
Y, X, Z = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz), indexing="ij")

fig3d = go.Figure()
for w in wastes:
    cid = w["id"]
    mask = (mat3d == cid)
    xs = X[mask].ravel()
    ys = Y[mask].ravel()
    zs = Z[mask].ravel()
    if xs.size == 0:
        # 仍放一條空 trace，避免 JS 對 trace index 失配
        xs, ys, zs = np.array([]), np.array([]), np.array([])

    text = [f"x={x}, y={y}, z={z}<br>代碼:{w['code']}<br>名稱:{w['name']}<br>廠區:{w['site']}<br>熱值:{w['kcal']:,} kcal/kg"
            for x, y, z in zip(xs, ys, zs)]

    fig3d.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        name=f"{w['code']} | {w['name']}",
        marker=dict(
            size=4,
            symbol="square",
            # 用同一色階：每類固定熱值，顏色由熱值決定（紅高藍低）
            color=np.full_like(xs, w["kcal"], dtype=float),
            colorscale="RdBu_r",
            cmin=float(kcal_lookup.min()),
            cmax=float(kcal_lookup.max()),
            opacity=0.85,
        ),
        text=text,
        hoverinfo="text",
        visible=True,
    ))

fig3d.update_layout(
    title="3D 點雲",
    scene=dict(
        xaxis=dict(title="X", tickmode="array",
                   tickvals=list(range(nx)), ticktext=[str(i) for i in range(nx)]),
        yaxis=dict(title="Y", tickmode="array",
                   tickvals=list(range(ny)), ticktext=[str(i) for i in range(ny)]),
        zaxis=dict(title="Depth(z)", tickmode="array",
                   tickvals=list(range(nz)), ticktext=[str(i) for i in range(nz)], autorange=False, range=[0, nz-1]),
        aspectmode="manual",
        aspectratio=dict(x=2.5, y=0.45, z=1.1),
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    height=800,
     paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e",
    font=dict(color="#e5e5e5"),
    xaxis=dict(
        gridcolor="#3c3c3c",
        zerolinecolor="#3c3c3c"
    ),
    yaxis=dict(
        gridcolor="#3c3c3c",
        zerolinecolor="#3c3c3c"
    ),
    scene_camera=dict(
        up=dict(x=0, y=2, z=1),      # z 軸朝上（預設）
        center=dict(x=0.5, y=0, z=-0.2),  # 看向資料中心
        eye=dict(x=2.0, y=1.4, z=0.8)
    )
)

# ============ 2D：先產生 z=0 的 heatmap；後續由 JS 依 checkbox+slider 重新計算 ============
def heat_and_text_for_layer(z_idx: int):
    layer = mat3d[:, :, z_idx]          # (ny,nx) class id
    heat = kcal_lookup[layer]           # (ny,nx) kcal

    text = np.empty((ny, nx), dtype=object)
    for yi in range(ny):
        for xi in range(nx):
            w = wastes[int(layer[yi, xi])]
            text[yi, xi] = (
                f"x={xi}, y={yi}, z={z_idx}<br>"
                f"代碼:{w['code']}<br>"
                f"名稱:{w['name']}<br>"
                f"廠區:{w['site']}<br>"
                f"熱值:{w['kcal']:,} kcal/kg"
            )
    return heat, text

heat0, text0 = heat_and_text_for_layer(0)
shapes = []
annotations = []

for s in segments:
    shapes.append(dict(
        type="rect",
        xref="x", yref="paper",
        x0=s["x0"] - 0.5,
        x1=s["x1"] - 0.5,
        y0=0,
        y1=1,
        line=dict(color="black", width=3),
        layer="above"
    ))

    annotations.append(dict(
        x=(s["x0"] + s["x1"] - 1) / 2,
        y=-0.08,
        xref="x",
        yref="paper",
        text=f"{s['door']}號門",
        showarrow=False,
        font=dict(size=16, color="#e5e5e5")
    ))

zones = [
    dict(name="A區", x0=6,  x1=10),
    dict(name="B區", x0=10, x1=14),
    dict(name="C區\n常用配方區", x0=14, x1=19),
    dict(name="D區", x0=19, x1=23),
]

for z in zones:
    annotations.append(dict(
        x=(z["x0"] + z["x1"]) / 2,
        y=0.5,
        xref="x",
        yref="paper",
        text=z["name"],
        showarrow=False,
        font=dict(size=20, color="#e5e5e5")
    ))

shapes.append(dict(
    type="rect",
    xref="x", yref="paper",
    x0=8.5, x1=9.5,
    y0=0.75, y1=0.95,
    fillcolor="rgba(0,0,0,0)",
    line=dict(color="red", width=3, dash="dash")
))

annotations.append(dict(
    x=9,
    y=0.85,
    xref="x",
    yref="paper",
    text="破碎機<br>出口",
    showarrow=False,
    font=dict(size=14, color="red")
))


fig2d = go.Figure(
    data=[go.Heatmap(
        z=heat0,
        x=list(range(nx)),
        y=list(range(ny)),
        colorscale="RdBu_r",
        zmin=float(kcal_lookup.min()),
        zmax=float(kcal_lookup.max()),
        colorbar=dict(title="kcal/kg"),
        text=text0,
        hoverinfo="text",
        # 讓 null/NaN 看起來是空的（後面 JS 會塞 null）
        connectgaps=False,
    )],
    layout=go.Layout(
        title="z 深度熱圖",
        xaxis=dict(title="X", tickmode="array",
                   tickvals=list(range(nx)), ticktext=[str(i) for i in range(nx)]),
        yaxis=dict(title="Y", tickmode="array",
                   tickvals=list(range(ny)), ticktext=[str(i) for i in range(ny)],
                   autorange="reversed"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
)

# 這裡保留 slider 外觀，但真正更新交給 JS（用 plotly_sliderchange 事件）
fig2d.update_layout(
    shapes=shapes,
    annotations=annotations,
    sliders=[dict(
        currentvalue=dict(prefix="Depth z = "),
        pad=dict(t=40),
        steps=[dict(label=str(zi), method="animate", args=[[str(zi)]]) for zi in range(nz)]
    )],
    height=800,
     paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e",
    font=dict(color="#e5e5e5"),
    xaxis=dict(
        gridcolor="#3c3c3c",
        zerolinecolor="#3c3c3c"
    ),
    yaxis=dict(
        gridcolor="#3c3c3c",
        zerolinecolor="#3c3c3c"
    )
)

# ============ 轉 HTML DIV ============
div3d = pio.to_html(fig3d, include_plotlyjs="cdn", full_html=False, div_id="view3d")
div2d = pio.to_html(fig2d, include_plotlyjs=False, full_html=False, div_id="view2d")

# ============ 把資料嵌到 HTML（mat3d + wastes + kcal lookup）===========
payload = {
    "ny": ny, "nx": nx, "nz": nz,
    "wastes": wastes,
    "kcal_lookup": kcal_lookup.tolist(),
    "mat3d": mat3d.astype(int).tolist(),   # 5x27x30
}

# checkbox HTML
checkbox_html = "\n".join(
    f'''
    <label class="cb">
      <input type="checkbox" class="matcb" value="{w["id"]}" checked>
      <span>{w["code"]}｜{w["name"]}（{w["kcal"]:,} kcal/kg / {w["site"]}）</span>
    </label>
    '''.strip()
    for w in wastes
)

html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>智慧儲坑儀表板</title>
  <style>
    body {{ font-family: sans-serif; margin: 0;background: #1e1e1e; }}
    .toolbar {{ padding: 10px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; border-bottom:1px solid #eee; background: #252526;}}
    .btn {{ padding: 8px 12px; border: 1px solid #ccc; background: #2d2d30;color: #e5e5e5; cursor:pointer; }}
    .btn.active {{ border-color:#ffffff; font-weight:600; }}
    .panel {{ padding: 10px; display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; background: #1e1e1e;}}
    .filters {{ min-width: 340px; border:1px solid #eee; padding:10px; border-radius:8px; background: #2d2d30;}}
    .filters h3 {{ margin:0 0 8px 0; font-size:14px; color: #ffffff;}}
    .cb {{ display:flex; gap:8px; align-items:flex-start; margin:6px 0; font-size:13px; line-height:1.2; color: #d4d4d4;}}
    .viewwrap {{ width: 100%; height: calc(100vh - 170px); background: #1e1e1e;}}
    #view3d, #view2d {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>

  <div class="toolbar">
    <button class="btn active" id="btn3d" onclick="showView('3d')">3D 點雲</button>
    <button class="btn" id="btn2d" onclick="showView('2d')">z 深度熱圖</button>
  </div>

  <div class="panel">
    <div class="filters">
      <h3>物料種類（多選）</h3>
      {checkbox_html}
      <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
        <button class="btn" onclick="selectAll(true)">全選</button>
        <button class="btn" onclick="selectAll(false)">全不選</button>
      </div>
      <div style="margin-top:10px; font-size:12px; color:#666;">
        說明：checkbox 會同時影響 3D 點雲與 2D 熱圖（未勾選類別在熱圖中會變成空白）。
      </div>
    </div>
    <div style="flex:1;min-width:320px; border:1px solid #eee; padding:10px; border-radius:8px; background:#2d2d30; color:#d4d4d4;">
      <div style="margin:0 0 8px 0; font-size:16px; color:#ffffff; font-weight:600;">
        下班次車次
      </div>

      <!-- 可捲動區 -->
      <div style="max-height:180px; overflow-y:auto; border:1px solid #3a3a3a; border-radius:6px; padding:8px; background:#252526;">
        <ul style="margin:0; padding-left:18px; font-size:15px; line-height:1.6;">
          <li>2026/02/06 下午 車牌 XXX-XXXX 廢棄物代碼 D-0202</li>
          <li>2026/02/06 下午 車牌 AAA-1234 廢棄物代碼 D-0301</li>
          <li>2026/02/07 上午 車牌 BBB-5678 廢棄物代碼 D-0101</li>
          <li>2026/02/07 上午 車牌 CCC-9999 廢棄物代碼 D-0402</li>
          <li>2026/02/07 下午 車牌 DDD-8888 廢棄物代碼 D-0202</li>
          <li>2026/02/07 下午 車牌 EEE-7777 廢棄物代碼 D-0301</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="viewwrap">
    <div id="wrap3d">{div3d}</div>
    <div id="wrap2d" style="display:none;">{div2d}</div>
  </div>

  <script>
    const DATA = {json.dumps(payload, ensure_ascii=False)};
    let currentZ = 0;

    function showView(which) {{
      const w3 = document.getElementById('wrap3d');
      const w2 = document.getElementById('wrap2d');
      const b3 = document.getElementById('btn3d');
      const b2 = document.getElementById('btn2d');

      if (which === '3d') {{
        w3.style.display = 'block';
        w2.style.display = 'none';
        b3.classList.add('active');
        b2.classList.remove('active');
        if (window.Plotly) Plotly.Plots.resize('view3d');
      }} else {{
        w3.style.display = 'none';
        w2.style.display = 'block';
        b2.classList.add('active');
        b3.classList.remove('active');
        if (window.Plotly) Plotly.Plots.resize('view2d');
      }}
    }}

    function getSelectedSet() {{
      const cbs = Array.from(document.querySelectorAll('.matcb'));
      const sel = new Set();
      cbs.forEach(cb => {{ if (cb.checked) sel.add(parseInt(cb.value)); }});
      return sel;
    }}

    function selectAll(on) {{
      Array.from(document.querySelectorAll('.matcb')).forEach(cb => cb.checked = on);
      applyFilter();
    }}

    // 更新 3D：每個類別一條 trace，直接切 visible
    function update3D(sel) {{
      const vis = DATA.wastes.map(w => sel.has(w.id));
      Plotly.restyle('view3d', {{visible: vis}});
    }}

    // 依選取類別 & z 層，重算 2D heat + hover text（未選類別 -> null）
    function compute2D(sel, zIdx) {{
      const ny = DATA.ny, nx = DATA.nx;
      const layer = DATA.mat3d;       // [y][x][z]
      const kcal = DATA.kcal_lookup;  // [classId]
      const wastes = DATA.wastes;

      const z = [];
      const text = [];
      for (let y=0; y<ny; y++) {{
        const rowZ = [];
        const rowT = [];
        for (let x=0; x<nx; x++) {{
          const cid = layer[y][x][zIdx];
          if (!sel.has(cid)) {{
            rowZ.push(null);
            rowT.push(`x=${{x}}, y=${{y}}, z=${{zIdx}}<br>(此類別未勾選)`);
          }} else {{
            const w = wastes[cid];
            rowZ.push(kcal[cid]);
            rowT.push(
                `x=${{x}}, y=${{y}}, z=${{zIdx}}<br>` +
                `代碼:${{w.code}}<br>` +
                `名稱:${{w.name}}<br>` +
                `廠區:${{w.site}}<br>` +
                `熱值:${{Number(w.kcal).toLocaleString()}} kcal/kg`
            );
          }}
        }}
        z.push(rowZ);
        text.push(rowT);
      }}
      return {{z, text}};
    }}

    function update2D(sel) {{
      const out = compute2D(sel, currentZ);
      Plotly.restyle('view2d', {{z: [out.z], text: [out.text], hoverinfo: ["text"]}}, [0]);
    }}

    function applyFilter() {{
      const sel = getSelectedSet();
      update3D(sel);
      update2D(sel);
    }}

    // 監聽 checkbox
    Array.from(document.querySelectorAll('.matcb')).forEach(cb => {{
      cb.addEventListener('change', applyFilter);
    }});

    // 監聽 Plotly slider（使用 plotly_sliderchange 事件取目前 z）
    const view2d = document.getElementById('view2d');
    view2d.on('plotly_sliderchange', function(e) {{
      const label = e.step && e.step.label ? e.step.label : "0";
      currentZ = parseInt(label);
      applyFilter();
    }});

    // 初始套用一次
    applyFilter();
  </script>

</body>
</html>
"""

with open("MER2dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Wrote dashboard.html")
