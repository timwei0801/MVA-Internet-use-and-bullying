import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.pyplot as plt
import os

# 設定中文字體
plt.rcParams['font.family'] = ['Arial Unicode MS']  # Mac OS 的通用中文字體
plt.rcParams['axes.unicode_minus'] = False


# 在程式碼開頭添加建立資料夾的函數
def create_output_directory(directory_name='output_figures'):
    """建立輸出圖片的目錄"""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"已建立 {directory_name} 目錄")
    return directory_name

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"成功讀取資料檔案，資料維度：{df.shape}")
        return df
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'")
        raise
    except Exception as e:
        print(f"讀取資料時發生錯誤：{str(e)}")
        raise

def preprocess_data(df):
    variables = [
        'q22_01_1', 'q22_02_1', 'q22_03_1', 'q22_04_1', 'q22_05_1',
        'q23_01_1', 'q23_02_1', 'q23_03_1', 'q23_04_1', 'q23_05_1',
        'q25_01_1', 'q25_02_1', 'q25_03_1', 'q25_04_1',
        'q26_01_1', 'q26_02_1', 'q26_03_1'
    ]
    
    # 檢查變數是否都存在
    missing_cols = [col for col in variables if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下變數在資料中不存在：{missing_cols}")
    
    # 檢查資料型態並轉換為數值型
    analysis_data = df[variables].apply(pd.to_numeric, errors='coerce')
    
    # 基本統計描述
    print("\n變數的基本統計資訊：")
    print(analysis_data.describe())
    
    # 遺漏值處理
    missing_stats = analysis_data.isnull().sum()
    if missing_stats.sum() > 0:
        print("\n遺漏值統計：")
        print(missing_stats[missing_stats > 0])
        analysis_data = analysis_data.fillna(analysis_data.mean())
    
    return analysis_data

def check_assumptions(data):
    kmo_all, kmo_model = calculate_kmo(data)
    print(f'KMO Score: {kmo_model:.3f}')
    
    chi_square_value, p_value = calculate_bartlett_sphericity(data)
    print(f'Bartlett test statistic: {chi_square_value:.3f}')
    print(f'Bartlett test p-value: {p_value:.3e}')
    
    return kmo_model, chi_square_value, p_value

def detailed_kmo_analysis(data):
    """執行詳細的KMO分析"""
    kmo_all, kmo_model = calculate_kmo(data)
    
    print("\nKMO 分析結果:")
    print(f"整體 KMO 評分: {kmo_model:.3f}")
    
    # KMO 評分解釋
    if kmo_model >= 0.9:
        print("KMO 評分解釋: 極佳 (Marvelous)")
    elif kmo_model >= 0.8:
        print("KMO 評分解釋: 優良 (Meritorious)")
    elif kmo_model >= 0.7:
        print("KMO 評分解釋: 中等 (Middling)")
    elif kmo_model >= 0.6:
        print("KMO 評分解釋: 普通 (Mediocre)")
    else:
        print("KMO 評分解釋: 不適合 (Unacceptable)")
    
    return kmo_all, kmo_model

def calculate_bartlett_sphericity(data):
    correlation_matrix = data.corr()
    n = data.shape[0]
    p = data.shape[1]
    chi_square = -(n - 1 - (2 * p + 5) / 6.) * np.log(np.linalg.det(correlation_matrix))
    df = p * (p - 1) / 2
    p_value = stats.chi2.sf(chi_square, df)
    return chi_square, p_value

def detailed_bartlett_analysis(data):
    """執行詳細的Bartlett球型檢定"""
    chi_square, p_value = calculate_bartlett_sphericity(data)
    
    print("\nBartlett's 球型檢定結果:")
    print(f"卡方值: {chi_square:.3f}")
    print(f"自由度: {(data.shape[1] * (data.shape[1] - 1)) / 2:.0f}")
    print(f"p值: {p_value:.3e}")
    print(f"檢定結果: {'顯著' if p_value < 0.05 else '不顯著'}")
    
    return chi_square, p_value

def perform_factor_analysis(data, n_factors=None):
    # 標準化數據
    data_standardized = (data - data.mean()) / data.std()
    
    # 計算相關矩陣
    corr_matrix = data_standardized.corr()
    
    # 初始特徵值分析
    fa_initial = FactorAnalyzer(rotation=None, n_factors=data.shape[1])
    fa_initial.fit(data_standardized)
    ev, v = fa_initial.get_eigenvalues()
    
    # 確定因素數量
    if n_factors is None:
        n_factors = sum(ev > 1)
    
    try:
        # 使用明確設定的參數執行因素分析
        fa = FactorAnalyzer(
            rotation='varimax',
            n_factors=n_factors,
            rotation_kwargs={'max_iter': 2000},
            impute='mean'  # 添加這個參數
        )
        
        # 使用標準化後的數據進行擬合
        fa.fit(data_standardized)
        
        # 獲取因素負荷量
        loadings = pd.DataFrame(
            fa.loadings_,
            columns=[f'Factor{i+1}' for i in range(n_factors)],
            index=data.columns
        )
        
        # 計算共同性
        communalities = pd.Series(
            fa.get_communalities(),
            index=data.columns,
            name='Communality'
        )
        
        # 計算特徵值和解釋變異量
        eigenvalues = pd.Series(ev[:n_factors], name='Eigenvalue')
        explained_variance = pd.Series(ev[:n_factors] / len(ev) * 100, name='Explained Variance %')
        
        print("\n特徵值和解釋變異量：")
        for i in range(n_factors):
            print(f"Factor {i+1}:")
            print(f"- 特徵值: {eigenvalues[i]:.3f}")
            print(f"- 解釋變異量: {explained_variance[i]:.2f}%")
        
        print("\n累積解釋變異量: {:.2f}%".format(explained_variance.sum()))
        
        return loadings, communalities, eigenvalues, explained_variance
        
    except Exception as e:
        print(f"執行因素分析時發生錯誤：{str(e)}")
        print("嘗試使用較小的因素數...")
        
        # 如果失敗，嘗試使用較小的因素數
        try:
            return perform_factor_analysis(data, n_factors=3)
        except:
            print("使用較小因素數仍然失敗")
            return None, None, None, None
        
        
def enhanced_factor_extraction(data, n_factors=None):
    """優化的因素萃取過程"""
    fa_initial = FactorAnalyzer(rotation=None)
    fa_initial.fit(data)
    ev, v = fa_initial.get_eigenvalues()
    
    # 計算累積解釋變異量
    total_var = sum(ev)
    cum_var = 0
    
    print("\n因素萃取結果:")
    for i, eigenval in enumerate(ev):
        if eigenval > 1:
            var_explained = (eigenval/total_var*100)
            cum_var += var_explained
            print(f"\nFactor {i+1}:")
            print(f"特徵值: {eigenval:.3f}")
            print(f"解釋變異量: {var_explained:.2f}%")
            print(f"累積解釋變異量: {cum_var:.2f}%")
    
    return ev, cum_var

def compare_rotation_methods(data, n_factors):
    """比較不同轉軸方法的結果"""
    rotation_methods = ['varimax', 'promax', 'oblimin']
    results = {}
    
    for method in rotation_methods:
        try:
            fa = FactorAnalyzer(rotation=method, n_factors=n_factors)
            fa.fit(data)
            
            # 獲取因素負荷量矩陣
            loadings = pd.DataFrame(
                fa.loadings_,
                columns=[f'Factor{i+1}' for i in range(n_factors)],
                index=data.columns
            )
            results[method] = loadings
            
            print(f"\n{method.capitalize()} 轉軸結果:")
            print(loadings)
            
            # 對於斜交轉軸方法，計算因素相關矩陣
            if method in ['promax', 'oblimin']:
                try:
                    # 確保因素相關矩陣維度正確
                    factor_corr = np.corrcoef(fa.transform(data).T)
                    factor_corr_df = pd.DataFrame(
                        factor_corr,
                        columns=[f'Factor{i+1}' for i in range(n_factors)],
                        index=[f'Factor{i+1}' for i in range(n_factors)]
                    )
                    print(f"\n{method.capitalize()} 因素相關矩陣:")
                    print(factor_corr_df)
                except Exception as corr_error:
                    print(f"計算{method}因素相關矩陣時發生錯誤：{str(corr_error)}")
                    
        except Exception as e:
            print(f"{method}轉軸過程中發生錯誤：{str(e)}")
            continue
    
    return results

def plot_scree(data, output_dir):
    fa = FactorAnalyzer()
    fa.fit(data)
    ev, v = fa.get_eigenvalues()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ev) + 1), ev)
    plt.title('碎石圖')
    plt.xlabel('因素數')
    plt.ylabel('特徵值')
    plt.axhline(y=1, color='r', linestyle='--')
    
    # 儲存圖片
    plt.savefig(os.path.join(output_dir, 'scree_plot.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"碎石圖已儲存至 {output_dir}/scree_plot.png")

def plot_factor_loadings(loadings, output_dir):
    plt.figure(figsize=(12, 8))
    
    # 設定熱圖參數
    sns.heatmap(loadings, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.3f',
                cbar_kws={'label': '因素負荷量'})
    
    plt.title('因素負荷量矩陣熱圖', pad=20)
    plt.xlabel('因素', labelpad=10)
    plt.ylabel('變數', labelpad=10)
    
    # 調整版面配置
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig(os.path.join(output_dir, 'factor_loadings_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"因素負荷量熱圖已儲存至 {output_dir}/factor_loadings_heatmap.png")

def calculate_and_save_factor_scores(data, loadings, output_dir):
    """計算因素分數並儲存結果"""
    # 標準化數據
    data_standardized = (data - data.mean()) / data.std()
    
    # 計算因素分數
    factor_scores = np.dot(data_standardized, loadings)
    
    # 轉換為DataFrame並加入有意義的列名
    factor_scores_df = pd.DataFrame(
        factor_scores,
        columns=['網路行為規範', '霸凌行為', '負面影響認知', '衝突容忍度']
    )
    
    # 儲存因素分數
    factor_scores_df.to_csv(os.path.join(output_dir, 'factor_scores.csv'), index=False)
    
    return factor_scores_df

def prepare_combined_dataset(factor_scores_df, original_data):
    """整合因素分數與原始資料"""
    # 選擇人口統計變數
    demographic_vars = ['q1', 'q2', 'q3', 'q4']  # 性別(q1)、年齡(q2)、教育程度(q4)
    
    # 選擇網路使用行為變數
    behavior_vars = [
        'q7',           # 上網時間
        'q9_1', 'q9_2', 'q9_3',  # 即時通訊軟體使用
        'q10_1', 'q10_2', 'q10_3',  # 社群媒體使用
        'q11_1', 'q11_2', 'q11_3'   # 影音平台使用
    ]
    
    # 合併資料
    selected_vars = demographic_vars + behavior_vars
    other_vars = original_data[selected_vars].copy()
    
    # 重新命名欄位以增加可讀性
    column_names = {
        'q1': '性別',
        'q2': '年齡',
        'q3': '職業',
        'q4': '教育程度',
        'q7': '上網時間',
        'q9_1': '即時通訊_LINE',
        'q9_2': '即時通訊_FB',
        'q9_3': '即時通訊_WeChat',
        'q10_1': '社群_Facebook',
        'q10_2': '社群_Instagram',
        'q10_3': '社群_Twitter',
        'q11_1': '影音_YouTube',
        'q11_2': '影音_愛奇藝',
        'q11_3': '影音_Netflix'
    }
    
    other_vars = other_vars.rename(columns=column_names)
    
    # 合併因素分數與其他變數
    combined_data = pd.concat([factor_scores_df, other_vars], axis=1)
    
    # 輸出整合後的資料資訊
    print("\n整合後的資料集資訊：")
    print(f"資料維度：{combined_data.shape}")
    print("\n包含的變數：")
    print(combined_data.columns.tolist())
    
    return combined_data

def plot_factor_analysis_results(factor_scores_df, output_dir):
    """產生因素分析結果的視覺化"""
    # 因素分數分布圖
    plt.figure(figsize=(12, 6))
    for col in factor_scores_df.columns:
        sns.kdeplot(data=factor_scores_df[col], label=col)
    plt.title('因素分數分布')
    plt.xlabel('因素分數')
    plt.ylabel('密度')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'factor_scores_distribution.png'))
    plt.close()
    
    # 因素間關係矩陣圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(factor_scores_df.corr(), 
                annot=True, 
                cmap='coolwarm',
                fmt='.2f')
    plt.title('因素間相關係數矩陣')
    plt.savefig(os.path.join(output_dir, 'factor_correlation_matrix.png'))
    plt.close()

def main():
    try:
        print("開始執行因素分析...")
        output_dir = create_output_directory()
        
        # 讀取和預處理資料
        df = read_data('processed_data_with_score.csv')
        analysis_data = preprocess_data(df)
        
        # 執行因素分析相關步驟
        kmo_all, kmo_model = detailed_kmo_analysis(analysis_data)
        chi_square, p_value = detailed_bartlett_analysis(analysis_data)
        ev, cum_var_ratio = enhanced_factor_extraction(analysis_data)
        plot_scree(analysis_data, output_dir)
        
        # 執行因素分析
        n_factors = sum(ev > 1)
        rotation_results = compare_rotation_methods(analysis_data, n_factors)
        loadings, communalities, eigenvalues, explained_variance = perform_factor_analysis(analysis_data, n_factors)
        plot_factor_loadings(loadings, output_dir)
        
        # 計算因素分數和整合資料
        factor_scores_df = calculate_and_save_factor_scores(analysis_data, loadings, output_dir)
        combined_data = prepare_combined_dataset(factor_scores_df, df)
        plot_factor_analysis_results(factor_scores_df, output_dir)
        
        # 儲存結果
        combined_data.to_csv(os.path.join(output_dir, 'combined_data_for_analysis.csv'), index=False)
        print("因素分析及後續處理完成，所有結果已儲存至output_figures資料夾")
        
        # 回傳結果
        return {
            'factor_scores': factor_scores_df,
            'combined_data': combined_data,
            'loadings': loadings,
            'communalities': communalities,
            'eigenvalues': eigenvalues,
            'explained_variance': explained_variance
        }
        
    except Exception as e:
        print(f"執行過程中發生錯誤：{str(e)}")
        return None

if __name__ == "__main__":
    main()