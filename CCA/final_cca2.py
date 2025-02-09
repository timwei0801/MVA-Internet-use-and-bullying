import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import statsmodels.multivariate.cancorr as cancorr
def setup_chinese_font():
    """設置支援中文的字體"""
    plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False

def plot_correlation_heatmap(data, title):
    """繪製相關係數熱圖"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_unified_weights(x_weights, y_weights, x_columns, y_columns, pair_number):
    """繪製統一的權重圖"""
    # 創建一個圖
    plt.figure(figsize=(15, 6))
    
    # 計算所有權重的最大和最小值，用於統一y軸範圍
    min_weight = min(np.min(x_weights), np.min(y_weights))
    max_weight = max(np.max(x_weights), np.max(y_weights))
    
    # 設置柱狀圖的位置
    x_positions = np.arange(len(x_columns))
    y_positions = np.arange(len(x_columns), len(x_columns) + len(y_columns))
    
    # 繪製柱狀圖
    plt.bar(x_positions, x_weights, color='steelblue', alpha=0.7)
    plt.bar(y_positions, y_weights, color='steelblue', alpha=0.7)
    
    # 設置刻度標籤
    all_positions = np.concatenate([x_positions, y_positions])
    all_labels = list(x_columns) + list(y_columns)
    
    plt.xticks(all_positions, all_labels, rotation=45, ha='right')
    
    # 設置y軸範圍
    plt.ylim(min_weight - 0.1, max_weight + 0.1)
    
    # 添加標題和標籤
    plt.title(f'網路使用行為與負面情緒變量權重 - 第{pair_number}對典型相關')
    plt.ylabel('Weights')
    
    # 添加分隔線
    plt.axvline(x=len(x_columns) - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 添加變量組標籤
    plt.text(len(x_columns) / 2 - 0.5, max_weight + 0.05, '網路使用行為', 
             horizontalalignment='center')
    plt.text(len(x_columns) + len(y_columns) / 2 - 0.5, max_weight + 0.05, 
             '網路負面情緒', horizontalalignment='center')
    
    # 調整布局
    plt.tight_layout()
    plt.show()

def main():
    # 載入數據
    df = pd.read_csv('processed_data_with_score2.csv')

    # 將數據轉為數值型，非數值設為 NaN
    X = df[['q5', 'q6', 'q7']].apply(pd.to_numeric, errors='coerce')
    Y = df[['q22_01_1', 'q22_02_1', 'q22_03_1', 'q22_04_1', 'q22_05_1',
            'q23_01_1', 'q23_02_1', 'q23_03_1', 'q23_04_1', 'q23_05_1',
            'q25_01_1', 'q25_02_1', 'q25_03_1', 'q25_04_1']].apply(pd.to_numeric, errors='coerce')

    # 填補缺失值
    X = X.fillna(X.mean())
    Y = Y.fillna(Y.mean())

    # 繪製原始變數的相關係數熱圖
    setup_chinese_font()
    plot_correlation_heatmap(X, "相關係數熱圖 (網路使用行為)")
    plot_correlation_heatmap(Y, "相關係數熱圖 (網路負面情緒)")

    # 標準化數據
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = scaler.fit_transform(Y)

    # 進行 CCA 分析
    cca = CCA(n_components=2)  # 取 2 對典型相關變量
    X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

    # 輸出典型相關係數
    corr1 = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    corr2 = np.corrcoef(X_c[:, 1], Y_c[:, 1])[0, 1]
    print(f"\n第一對典型相關係數: {corr1:.3f}")
    print(f"第二對典型相關係數: {corr2:.3f}")

    # 輸出典型變量的權重 (Weights)
    print("\n=== X 變量的權重 (Weights) ===")
    for i, col in enumerate(X.columns):
        print(f"{col}: 第一對 = {cca.x_weights_[i, 0]:.4f}, 第二對 = {cca.x_weights_[i, 1]:.4f}")

    print("\n=== Y 變量的權重 (Weights) ===")
    for i, col in enumerate(Y.columns):
        print(f"{col}: 第一對 = {cca.y_weights_[i, 0]:.4f}, 第二對 = {cca.y_weights_[i, 1]:.4f}")

    # 繪製典型相關變量的散點圖
    plt.figure(figsize=(10, 6))
    plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.5, label=f"第一對典型相關 (r = {corr1:.3f})")
    plt.scatter(X_c[:, 1], Y_c[:, 1], alpha=0.5, label=f"第二對典型相關 (r = {corr2:.3f})")
    plt.xlabel("網路使用行為典型變量")
    plt.ylabel("網路負面情緒典型變量")
    plt.title("典型相關分析散點圖")
    plt.legend()
    plt.grid()
    plt.show()

    # 使用新的統一權重圖函數
    plot_unified_weights(
        cca.x_weights_[:, 0], 
        cca.y_weights_[:, 0], 
        X.columns, 
        Y.columns, 
        "一"
    )
    
    plot_unified_weights(
            cca.x_weights_[:, 1], 
            cca.y_weights_[:, 1], 
            X.columns, 
            Y.columns, 
            "二"
        )

if __name__ == "__main__":
    main()