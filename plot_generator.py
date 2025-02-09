import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from cartopy import crs, feature
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
import matplotlib.colors as mcolors
import os

output_folder = 'plot_output'
os.makedirs(output_folder, exist_ok=True)

# 性別分布圖
labels = ['Male', 'Female']
sizes = [61.5, 38.5]
colors = ['#1f77b4', '#17becf']

fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
       startangle=90, pctdistance=0.85, labeldistance=1.1, textprops={'fontsize': 12})
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal')
plt.title('Gender Distribution', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'gender_distribution_3d.png'), dpi=300)


# 出生年份分布圖
years = ['Before 1960', '1961-1970', '1971-1980', '1981-1990', 'After 1990']
percentages = np.array([20.7, 21.5, 27.6, 29.7, 0.6])

fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(5)
colors = cm.Blues(np.linspace(0.3, 1.0, 5))

for i, p in enumerate(percentages):
    ax.bar(years, p, bottom=bottom, color=colors[i])
    bottom += p
    ax.text(i, bottom[i]-p/2, f'{p}%', ha='center', fontsize=12)

plt.ylabel('Percentage', fontsize=14)  
plt.title('Birth Year Distribution', fontsize=16)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'birth_year_distribution_stacked.png'), dpi=300)


# 讀取台灣地圖資料
shapefile_path = "taiwan_map/COUNTY_MOI_1130718.shp"  # 確保這是您的本地路徑
taiwan_map = gpd.read_file(shapefile_path)

# 假設人口比例數據儲存在 data 字典中
data = {
    '臺北市': 37.1, '桃園市': 37.1, '新竹縣': 37.1, '苗栗縣': 26.7,
    '臺中市': 26.7, '彰化縣': 26.7, '南投縣': 26.7, '雲林縣': 26.7,
    '嘉義縣': 30.1, '臺南市': 30.1, '高雄市': 30.1, '屏東縣': 30.1,
    '宜蘭縣': 5.2, '花蓮縣': 5.2, '臺東縣': 5.2, '澎湖縣': 0.9,
    '金門縣': 0.9, '連江縣': 0.9
}

# 將數據加入地圖
taiwan_map['percentage'] = taiwan_map['COUNTYNAME'].map(data)

# 設置顏色映射
cmap = cm.YlOrRd
norm = Normalize(vmin=taiwan_map['percentage'].min(), vmax=taiwan_map['percentage'].max())

# 繪製地圖
fig, ax = plt.subplots(1, figsize=(10, 10))  
taiwan_map.plot(column='percentage', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8')

# 設置地圖邊界，放大台灣地區
ax.set_xlim(119.8, 122.2)  
ax.set_ylim(21.8, 25.5)

# 加入缺失縣市的數據
taiwan_map.loc[taiwan_map['COUNTYNAME'] == '新北市', 'percentage'] = 37.1
taiwan_map.plot(column='percentage', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8')
taiwan_map.loc[taiwan_map['COUNTYNAME'] == '嘉義市', 'percentage'] = 30.1
taiwan_map.plot(column='percentage', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8')
missing_counties = ['基隆市', '新竹市']
for county in missing_counties:
    taiwan_map.loc[taiwan_map['COUNTYNAME'] == county, 'percentage'] = 37.1
taiwan_map.plot(column='percentage', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8')


# 加入圖例
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.03)
cbar.set_label('Population Percentage (%)', fontsize=14)

# 增加表格
table_data = [    
    ['Region', 'Percentage'],
    ['North', '37.1%'], 
    ['West', '26.7%'], 
    ['South', '30.1%'],
    ['East', '5.2%'], 
    ['Others', '0.9%']
]

# 調整表格位置避免遮擋
plt.subplots_adjust(right=0.7)  

table = plt.table(cellText=table_data, colLabels=None, cellLoc='right',
                  bbox=[1.03, 0.07, 0.4, 0.15], edges='open')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2) 

plt.title("Taiwan Regional Distribution", fontsize=20) 
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'regional_distribution_map.png'), dpi=300, bbox_inches='tight') 
plt.show()


# 設置數據
usage_data = {
    '0-3 hours': 41.1,
    '3-6 hours': 40.7,
    'More than 6 hours': 18.2
}

# 創建圖表
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

# 設置色彩映射
colors = ['#2C699A', '#48A9A6', '#E4DFDA']
values = list(usage_data.values())
labels = list(usage_data.keys())

# 計算角度
angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)
width = 2*np.pi/len(values)

# 繪製放射狀圖
bars = ax.bar(angles, values, width=width, bottom=20,
             color=colors, alpha=0.7)

# 添加標籤
for angle, value, label in zip(angles, values, labels):
    # 外圈百分比標籤
    ax.text(angle, value + 25, f'{value}%',
            ha='center', va='center')
    # 內圈類別標籤
    ax.text(angle, 15, label,
            ha='center', va='center',
            rotation=angle*180/np.pi - 90,
            rotation_mode='anchor')

# 設置圖表樣式
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rgrids([])
ax.set_thetagrids([])
ax.set_ylim(0, 100)

# 添加標題
plt.title('Daily Internet Usage Distribution', pad=20, fontsize=16)

# 添加圓形邊框
circle = Circle((0, 0), 20, transform=ax.transData._b,
               facecolor='white', edgecolor='gray',
               linewidth=1, zorder=0)
ax.add_patch(circle)

# 微調佈局
plt.tight_layout()

# 儲存圖片
plt.savefig(os.path.join(output_folder,'internet_usage_radial.png'), dpi=300, bbox_inches='tight')
plt.show()
