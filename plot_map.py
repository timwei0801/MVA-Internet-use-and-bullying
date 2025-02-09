import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# 設定輸出資料夾和字體 
output_folder = 'plot_output'
os.makedirs(output_folder, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 創建3D圖表
fig = plt.figure(figsize=(15, 20))
ax = fig.add_subplot(111, projection='3d')

# 讀取台灣地圖
taiwan_map = gpd.read_file("taiwan_map/COUNTY_MOI_1130718.shp")

# 定義區域顏色
region_colors = {
    '北部': '#4A225D',  # 深紫色
    '中部': '#B4432B',  # 紅色
    '南部': '#E85D04',  # 橙色
    '東部': '#FFF3B0',  # 淺黃色
    '其他': '#CCCCCC'   # 灰色
}

# 定義縣市所屬區域
county_regions = {
    '臺北市': '北部', '新北市': '北部', '基隆市': '北部', '桃園市': '北部', 
    '新竹市': '北部', '新竹縣': '北部', '臺中市': '中部', '苗栗縣': '中部', 
    '彰化縣': '中部', '南投縣': '中部', '雲林縣': '中部', '高雄市': '南部', 
    '臺南市': '南部', '嘉義市': '南部', '嘉義縣': '南部', '屏東縣': '南部',
    '宜蘭縣': '東部', '花蓮縣': '東部', '臺東縣': '東部', '澎湖縣': '其他', 
    '金門縣': '其他', '連江縣': '其他'
}

# 設定3D視角
ax.view_init(elev=20, azim=45)
ax.set_box_aspect([2, 3, 1])

# 繪製基礎地圖
for idx, row in taiwan_map.iterrows():
    poly = row.geometry
    if poly.geom_type == 'Polygon':
        x, y = poly.exterior.xy
        ax.plot_trisurf(x, y, np.zeros_like(x), 
                       color=region_colors.get(county_regions.get(row['COUNTYNAME'], '其他')),
                       alpha=0.8)

# 定義各區域的數據
locations = {
    '北部': {
        'coords': (121.5, 24.9),
        'values': {
            '性別': [64.7, 35.3],
            '網路': [39.2, 42.2, 18.6],
            '年齡': [20.7, 21.5, 27.6, 29.7, 0.6]
        }
    },
    '中部': {
        'coords': (120.7, 24.1),
        'values': {
            '性別': [55.9, 44.1],
            '網路': [45.3, 38.4, 16.3],
            '年齡': [21.2, 22.4, 24.6, 31.2, 0.6]
        }
    },
    '南部': {
        'coords': (120.3, 22.6),
        'values': {
            '性別': [61.4, 38.6],
            '網路': [41.2, 40.5, 18.3],
            '年齡': [20.8, 21.8, 24.2, 32.2, 1.0]
        }
    },
    '東部': {
        'coords': (121.4, 23.5),
        'values': {
            '性別': [68.6, 31.4],
            '網路': [54.3, 31.4, 14.3],
            '年齡': [22.9, 20.0, 28.6, 28.6, 0.0]
        }
    }
}

# 設定長條圖參數
bar_colors = {
    '性別': ['#1f77b4', '#aec7e8'],
    '網路': ['#2ecc71', '#e67e22', '#2c3e50'],
    '年齡': ['#ff9999', '#ff7f7f', '#ff6666', '#ff4d4d', '#ff3333']
}

# 定義長條圖尺寸參數
bar_width = 0.04
bar_depth = 0.04
z_scale = 0.01
spacing = 0.15

# 繪製3D長條圖
for region, data in locations.items():
    x, y = data['coords']
    
    # 繪製性別數據
    values_gender = data['values']['性別']
    for i, (value, color) in enumerate(zip(values_gender, bar_colors['性別'])):
        x_pos = x + i * bar_width * 3
        ax.bar3d(x_pos, y, 0,
                bar_width, bar_depth, value * z_scale,
                color=color, alpha=0.8)
        ax.text(x_pos, y, value * z_scale + 0.02,
                f'{value}%',
                ha='center', va='bottom', fontsize=8)
    
    # 繪製網路使用時間
    values_internet = data['values']['網路']
    for i, (value, color) in enumerate(zip(values_internet, bar_colors['網路'])):
        x_pos = x + (i + 3) * bar_width * 3
        ax.bar3d(x_pos, y, 0,
                bar_width, bar_depth, value * z_scale,
                color=color, alpha=0.8)
        ax.text(x_pos, y, value * z_scale + 0.02,
                f'{value}%',
                ha='center', va='bottom', fontsize=8)
    
    # 繪製年齡分布
    values_age = data['values']['年齡']
    for i, (value, color) in enumerate(zip(values_age, bar_colors['年齡'])):
        x_pos = x + (i + 7) * bar_width * 3
        ax.bar3d(x_pos, y, 0,
                bar_width, bar_depth, value * z_scale,
                color=color, alpha=0.8)
        ax.text(x_pos, y, value * z_scale + 0.02,
                f'{value}%',
                ha='center', va='bottom', fontsize=8)

# 設定圖例標籤
legend_labels = {
    '性別': ['男性', '女性'],
    '網路': ['0-3小時', '3-6小時', '6小時以上'],
    '年齡': ['60歲以上', '51-60歲', '41-50歲', '31-40歲', '21-30歲']
}

# 添加圖例
legend_elements = []
for category, colors in bar_colors.items():
    for color, label in zip(colors, legend_labels[category]):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8, label=label))
ax.legend(handles=legend_elements, 
         loc='upper right',
         bbox_to_anchor=(1.15, 1),
         title='類別說明')

# 設定標題與關閉座標軸
plt.title('台灣地區人口特徵分布（3D視圖）', fontsize=16, pad=20)
ax.set_axis_off()

# 添加資料來源註解
plt.figtext(0.02, 0.02, '資料來源：台灣傳播調查資料庫(2021)', fontsize=10)

# 調整版面配置並保存
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'taiwan_3d_distribution.png'), 
            dpi=300, bbox_inches='tight')
plt.show()