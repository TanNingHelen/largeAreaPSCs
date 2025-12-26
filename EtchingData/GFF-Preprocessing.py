import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from sklearn import preprocessing

warnings.filterwarnings('ignore')


def enhanced_preprocessing(df):
    """改进的数据预处理"""
    df1 = df.copy()
    df1['Original_GFF'] = df1['GFF']
    df1['Filled_GFF'] = (df1['GFF'] == 0).astype(int)

    print(f"数据统计:")
    print(f"总样本数: {len(df1)}")
    print(f"GFF不为0的样本数: {((df1['GFF'] != 0) & (df1['GFF'].notnull())).sum()}")
    print(f"GFF为0的样本数: {(df1['GFF'] == 0).sum()}")

    # 修正列名
    df1 = df1.rename(columns={
        'HTL-Additive': 'HTL-Addictive',
        'ETL-Additive': 'ETL-Addictive',
        'Precursor_Solution_Additive': 'Precursor_Solution_Addictive'
    })

    # 创建label_mappings目录
    os.makedirs("label_mappings", exist_ok=True)

    # 标签编码
    categorical_cols = ['Structure', 'HTL', 'HTL-2', 'HTL_Passivator', 'HTL-Addictive',
                        'ETL', 'ETL-2', 'ETL_Passivator', 'ETL-Addictive',
                        'Metal_Electrode', 'Glass', 'Precursor_Solution',
                        'Precursor_Solution_Addictive', 'Deposition_Method',
                        'Antisolvent', 'Type', 'brand']

    # 存储所有映射关系
    label_mappings = {}

    for col in categorical_cols:
        if col in df1.columns:
            lbl = preprocessing.LabelEncoder()
            df1[col] = lbl.fit_transform(df1[col].astype('str'))

            # 保存映射关系
            mapping = pd.DataFrame({
                'Original': lbl.classes_,
                'Encoded': range(len(lbl.classes_))
            })
            label_mappings[col] = mapping

            # 保存到CSV
            mapping.to_csv(f"label_mappings/{col}_mapping.csv", index=False)
            print(f"已保存 {col} 的映射关系到: label_mappings/{col}_mapping.csv")

    # 保存所有映射关系的汇总表
    mapping_summary = []
    for col, mapping_df in label_mappings.items():
        temp_df = mapping_df.copy()
        temp_df['Feature'] = col
        mapping_summary.append(temp_df)

    full_mapping = pd.concat(mapping_summary, ignore_index=True)
    full_mapping = full_mapping[['Feature', 'Original', 'Encoded']]  # 调整列顺序
    full_mapping.to_csv("label_mappings/full_mapping_summary.csv", index=False)

    print(f"\n=== 映射文件生成完成 ===")
    print(f"总映射特征数量: {len(label_mappings)}")
    print(f"总映射条目数量: {len(full_mapping)}")
    print(f"完整映射汇总已保存到: label_mappings/full_mapping_summary.csv")

    # 打印Deposition_Method的映射详情
    if 'Deposition_Method' in label_mappings:
        depo_mapping = label_mappings['Deposition_Method']
        print(f"\nDeposition_Method 映射详情:")
        print(f"唯一值数量: {len(depo_mapping)}")
        for _, row in depo_mapping.iterrows():
            print(f"  {row['Original']} -> {row['Encoded']}")

    return df1


def analyze_feature_importance(df):
    """分析特征重要性"""
    valid_data = df[(df['GFF'] != 0) & (df['GFF'].notnull())].copy()

    if len(valid_data) < 10:
        return []

    X = valid_data.drop(columns=['GFF', 'Original_GFF', 'Filled_GFF'], errors='ignore')
    y = valid_data['GFF']

    # 只选择数值型特征进行相关性分析
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    correlations = []
    for col in numeric_cols:
        corr = np.corrcoef(X[col], y)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))

    # 按相关性绝对值排序
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\n特征与GFF的相关性（前15）:")
    for i, (col, corr) in enumerate(correlations[:15], 1):
        print(f"  {i:2d}. {col:30} | 相关性: {corr:.4f}")

    # 返回相关性最高的特征
    return [col for col, _ in correlations[:15]]


def robust_cross_validation(X, y, model, n_splits=5):
    """稳健的交叉验证"""
    try:
        kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=42)
        mae_scores = []
        r2_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练模型
            model.fit(X_train_scaled, y_train)

            # 预测
            y_pred = model.predict(X_test_scaled)

            # 计算指标
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mae_scores.append(mae)
            r2_scores.append(r2)

        return np.mean(mae_scores), np.std(mae_scores), np.mean(r2_scores), np.std(r2_scores)

    except Exception as e:
        print(f"交叉验证出错: {e}")
        return np.inf, 0, -np.inf, 0


def evaluate_regression_models(df, selected_features):
    """评估回归模型性能"""
    valid_data = df[(df['GFF'] != 0) & (df['GFF'].notnull())].copy()

    if len(valid_data) < 20:
        print("有效数据不足，无法进行可靠评估")
        return None, None

    X = valid_data[selected_features]
    y = valid_data['GFF']

    # 定义模型
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        print(f"评估 {name}...")
        mae, mae_std, r2, r2_std = robust_cross_validation(X, y, model)
        results[name] = {
            'MAE': mae,
            'MAE_std': mae_std,
            'R2': r2,
            'R2_std': r2_std,
            'model': model
        }
        print(f"  {name}: MAE = {mae:.3f} ± {mae_std:.3f}, R² = {r2:.3f} ± {r2_std:.3f}")

    # 选择最佳模型
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
        best_result = results[best_model_name]

        print(f"\n最佳模型: {best_model_name}")
        print(f"MAE: {best_result['MAE']:.3f} ± {best_result['MAE_std']:.3f}")
        print(f"R²: {best_result['R2']:.3f} ± {best_result['R2_std']:.3f}")

        return best_model_name, results[best_model_name]['model']

    return None, None


def impute_with_best_model(df, selected_features, best_model):
    """使用最佳模型进行填充"""
    df_result = df.copy()

    # 分离已知数据和待填充数据
    known_data = df_result[(df_result['GFF'] != 0) & (df_result['GFF'].notnull())].copy()
    unknown_data = df_result[df_result['GFF'] == 0].copy()

    if len(unknown_data) == 0:
        print("没有需要填充的数据")
        return df_result

    # 准备训练数据
    X_train = known_data[selected_features]
    y_train = known_data['GFF']

    # 准备预测数据
    X_pred = unknown_data[selected_features]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # 训练模型
    best_model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = best_model.predict(X_pred_scaled)

    # 更新GFF值
    df_result.loc[df_result['GFF'] == 0, 'GFF'] = y_pred

    print(f"成功填充了 {len(unknown_data)} 个GFF为0的值")
    print(f"填充值的统计: 最小值={y_pred.min():.2f}, 最大值={y_pred.max():.2f}, 均值={y_pred.mean():.2f}")

    return df_result


def main():
    # 读取数据
    df = pd.read_excel(r"BandgapDone2.xlsx")

    # 预处理
    df_processed = enhanced_preprocessing(df)

    # 分析特征重要性
    selected_features = analyze_feature_importance(df_processed)

    if not selected_features:
        print("无法选择有效特征，使用简单填充")
        # 使用中位数填充
        valid_gff = df_processed[df_processed['GFF'] != 0]['GFF']
        median_gff = valid_gff.median()
        df_processed.loc[df_processed['GFF'] == 0, 'GFF'] = median_gff
        print(f"使用中位数填充: {median_gff:.2f}")
    else:
        # 评估模型
        best_model_name, best_model = evaluate_regression_models(df_processed, selected_features)

        if best_model is not None:
            # 使用最佳模型填充
            df_processed = impute_with_best_model(df_processed, selected_features, best_model)
        else:
            # 回退到中位数填充
            valid_gff = df_processed[df_processed['GFF'] != 0]['GFF']
            median_gff = valid_gff.median()
            df_processed.loc[df_processed['GFF'] == 0, 'GFF'] = median_gff
            print(f"模型评估失败，使用中位数填充: {median_gff:.2f}")

    # 后处理
    cols_to_drop = ['Sn', 'Rb']
    cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
    df_processed.drop(columns=cols_to_drop, inplace=True)

    # 保存未去除行的数据
    df_processed.to_excel(r"FinalDataAll.xlsx", index=False)
    print("未去除行的数据已保存到 FinalDataAll.xlsx")

    # 过滤激光特征全为0的行
    laser_cols = ['P1Wavelength(nm)', 'P2Wavelength(nm)', 'P3Wavelength(nm)',
                  'total_scribing_line_width(μm)',
                  'P1Width(μm)', 'P2Width(μm)', 'P3Width(μm)']
    laser_cols = [col for col in laser_cols if col in df_processed.columns]

    # 先将这些列转换为数值型
    for col in laser_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')


        all_zero_mask = (df_processed[laser_cols] == 0).all(axis=1)

        df_filtered = df_processed[~all_zero_mask]
        removed_count = all_zero_mask.sum()
        print(f"过滤掉了 {removed_count} 行激光特征全为0的数据")

        # 保存过滤后的数据
        df_filtered.to_excel(r"FinalData2.xlsx", index=False)
        print(f"过滤后的数据已保存到 FinalData2.xlsx，剩余 {len(df_filtered)} 行数据")
    else:
        # 如果没有激光特征列，直接保存
        df_processed.to_excel(r"FinalData2.xlsx", index=False)
        print("没有激光特征列，数据已保存到 FinalData2.xlsx")

    print("\n数据处理完成！")
    print(f"- 完整数据（未去除行）: FinalDataAll.xlsx")
    print(f"- 过滤后数据: FinalData2.xlsx")


if __name__ == "__main__":
    main()