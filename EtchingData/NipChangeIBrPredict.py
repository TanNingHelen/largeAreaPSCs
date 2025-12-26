import os
import joblib
import pandas as pd
import numpy as np
import warnings
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# 导入Column Splitting2.py中的函数
try:
    from Column_Splitting2 import get_element_ratio

    print("✅ 成功导入Column_Splitting2.py中的get_element_ratio函数")
except ImportError as e:
    print(f"❌ 导入Column_Splitting2.py失败: {e}")
    print("将使用内置的简化版本")


    # 如果导入失败，使用简化版本
    def get_element_ratio(composition):
        """
        简化版的元素比例解析函数
        """
        elements = {'Cs': 'A', 'MA': 'A', 'FA': 'A', 'Rb': 'A', 'Pb': 'B', 'Sn': 'B', 'I': 'X', 'Br': 'X', 'Cl': 'X'}
        element_ratio = {key: 0 for key in elements.keys()}

        # 简化解析逻辑
        try:
            if 'FA' in composition and 'MA' in composition:
                # 从化学式中提取FA和MA的比例
                fa_match = composition.split('FA')[-1].split(')')[0].split('MA')[0]
                ma_match = composition.split('MA')[-1].split(')')[0]

                try:
                    fa_ratio = float(fa_match) if fa_match.replace('.', '').isdigit() else 0.95
                    ma_ratio = float(ma_match) if ma_match.replace('.', '').isdigit() else 0.05
                except:
                    fa_ratio = 0.95
                    ma_ratio = 0.05
            else:
                fa_ratio = 0.95
                ma_ratio = 0.05

            # 设置默认值
            element_ratio['FA'] = fa_ratio
            element_ratio['MA'] = ma_ratio
            element_ratio['Cs'] = 0.05
            element_ratio['I'] = 0.98
            element_ratio['Br'] = 0.02
            element_ratio['Cl'] = 0.0
            element_ratio['Pb'] = 1.0

        except Exception as e:
            print(f"解析钙钛矿化学式失败: {e}")

        return element_ratio


def parse_perovskite_composition(composition):
    """
    解析钙钛矿化学式并返回元素比例 - 使用导入的get_element_ratio函数
    """
    try:
        # 先处理一些常见的格式问题
        composition = composition.replace('l', 'I')  # 修正小写l为大写I

        # 使用导入的get_element_ratio函数进行解析
        ratio = get_element_ratio(composition)

        # 确保所有必需的元素都有值
        required_elements = ['Cs', 'MA', 'FA', 'I', 'Br', 'Cl', 'Pb']
        for element in required_elements:
            if element not in ratio:
                ratio[element] = 0.0

        return ratio
    except Exception as e:
        print(f"解析钙钛矿化学式失败: {composition}, 错误: {e}")
        # 返回默认值
        return {'Cs': 0.0, 'MA': 0.0, 'FA': 0.0, 'I': 0.0, 'Br': 0.0, 'Cl': 0.0, 'Pb': 1.0}


def encode_categorical_features(df, mapping_df):
    """
    对分类特征进行编码
    """
    encoded_df = df.copy()
    categorical_features = [
        'Structure', 'HTL', 'HTL-2', 'HTL_Passivator', 'HTL-Addictive',
        'ETL', 'ETL-2', 'ETL_Passivator', 'ETL-Addictive',
        'Metal_Electrode', 'Glass', 'Precursor_Solution',
        'Precursor_Solution_Addictive', 'Deposition_Method',
        'Antisolvent', 'Type', 'brand'
    ]

    for feature in categorical_features:
        if feature in encoded_df.columns:
            feature_mapping = mapping_df[mapping_df['Feature'] == feature]
            if len(feature_mapping) > 0:
                mapping_dict = dict(zip(feature_mapping['Original'], feature_mapping['Encoded']))
                original_value = encoded_df[feature].iloc[0]

                if original_value == '' or pd.isna(original_value):
                    empty_mapping = feature_mapping[feature_mapping['Original'].isna()]
                    encoded_value = empty_mapping['Encoded'].iloc[0] if len(empty_mapping) > 0 else 0
                else:
                    encoded_value = mapping_dict.get(original_value, 0)

                encoded_df[feature] = encoded_value
            else:
                encoded_df[feature] = 0

    return encoded_df


def predict_i_br_combinations():
    """
    基于给定的实验数据，预测不同I和Br组合的PCE
    """
    # 模型信息
    MODELS_INFO = {
        "RandomForest": ("models/best_randomforest_model.pkl", 0.8616),
        "XGBoost": ("models/best_xgboost_model.pkl", 0.8835),
        "LightGBM": ("models/best_lgbm_model.pkl", 0.8630)
    }

    # 1. 加载模型
    print("=== 加载集成模型 ===")
    models = {}
    r2_values = {}

    for name, (path, r2) in MODELS_INFO.items():
        try:
            model = joblib.load(path)
            models[name] = model
            r2_values[name] = r2
            print(f"✅ {name}模型加载成功! (测试集R²: {r2})")
        except Exception as e:
            print(f"❌ {name}模型加载失败: {e}")

    if not models:
        print("❌ 没有成功加载任何模型")
        return None

    # 计算权重
    total_r2 = sum(r2_values.values())
    weights = {name: r2 / total_r2 for name, r2 in r2_values.items()}

    print(f"\n📊 模型权重分配 (基于测试集R²):")
    for name, weight in weights.items():
        print(f"   {name}: {weight:.4f} ({weight * 100:.2f}%)")

    # 2. 加载映射文件
    try:
        mapping_df = pd.read_csv('label_mappings/full_mapping_summary.csv')
        print("✅ 映射文件加载成功")
    except Exception as e:
        print(f"❌ 映射文件加载失败: {e}")
        return None

    # 3. 基础实验数据（基于您提供的代码）
    base_data = {
        'Structure': 'p-i-n',
        'HTL': 'NiOx',
        'HTL-2': 'Me-4PACz',
        'HTL_Passivator': '',
        'HTL-Addictive': 'DMPU',
        'ETL': 'C60',
        'ETL-2': 'SnO2',
        'ETL_Passivator': '',
        'ETL-Addictive': '',
        'Metal_Electrode': 'Cu',
        'Glass': 'FTO',
        'Perovskite': '(FA0.98MA0.02)0.95Cs0.05Pb(I0.98Br0.02)3',
        'Active_Area': 12.96,
        'Precursor_Solution': 'DMF:NMP (7:1)',
        'Precursor_Solution_Addictive': 'PbI2+MACI',
        'Deposition_Method': 'blade-coating',
        'Antisolvent': '',
        'Annealing_Temperature1': 120,
        'Annealing_Time1': 25,
        'Annealing_Temperature2': 0,
        'Annealing_Time2': 0,
        'P1Wavelength(nm)': 532,
        'P2Wavelength(nm)': 532,
        'P3Wavelength(nm)': 532,
        'total_scribing_line_width(μm)': 235,
        'P1Width(μm)': 40,
        'P2Width(μm)': 65,
        'P3Width(μm)': 40,
        'GFF': 95.36,
        'Type': 'Series',
        'submodule_number': 6,
        'P1Scan_Velocity(mm/s)': 4000,
        'P1etching_frequency(kHz)': 500,
        'P1Spot Size(μm)': 40,
        'P1etching_Power(W)': 0,
        'P1etching_Power_percentage(%)': 40,
        'P2Scan_Velocity': 2000,
        'P2etching_frequency(kHz)': 500,
        'P2Spot Size(μm)': 40,
        'P2etching_Power(W)': 0,
        'P2etching_Power_percentage(%)': 10,
        'P3Scan_Velocity': 2000,
        'P3etching_frequency(kHz)': 500,
        'P3Spot Size(μm)': 40,
        'P3etching_Power(W)': 0,
        'P3etching_Power_percentage(%)': 9,
        'P1_P2Scribing_Spacing(μm)': 45,
        'P2_P3Scribing_Spacing(μm)': 45,
        'brand': '',
        'Cs': 0.05,
        'MA': 0.02,
        'FA': 0.93,
        'I': 0.98,  # 这是我们要替换的列
        'Br': 0.02,  # 这是我们要替换的列
        'Cl': 0.0,
        'Bandgap': 1.0  # 初始值，会根据实际计算更新
    }

    # 4. 生成I和Br的组合
    print("\n🔬 生成I和Br组合...")
    # I和Br的比例应该满足 I + Br + Cl = 1，这里我们假设Cl固定为0
    i_values = np.arange(0.5, 1.0, 0.05)  # I从0.5到0.95
    br_values = np.arange(0.05, 0.5, 0.05)  # Br从0.05到0.45

    combinations = []
    for i_val in i_values:
        for br_val in br_values:
            if i_val + br_val <= 1.0:  # 确保I + Br <= 1
                cl_val = 1.0 - i_val - br_val  # Cl的比例
                if cl_val >= 0:  # 确保Cl不为负
                    combinations.append({
                        'I': round(i_val, 2),
                        'Br': round(br_val, 2),
                        'Cl': round(cl_val, 2)
                    })

    print(f"生成了 {len(combinations)} 个I和Br组合")

    # 5. 对每个组合进行预测
    results = []
    print(f"\n🎯 开始对 {len(combinations)} 个组合进行预测...")

    for i, combo in enumerate(combinations):
        # 创建新样本
        new_sample_data = base_data.copy()

        # 更新I和Br的值
        new_sample_data['I'] = combo['I']
        new_sample_data['Br'] = combo['Br']
        new_sample_data['Cl'] = combo['Cl']

        # 更新钙钛矿组成
        new_sample_data[
            'Perovskite'] = f'(FA{base_data["FA"]:.2f}MA{base_data["MA"]:.2f}){base_data["Cs"]:.2f}CsPb(I{combo["I"]:.2f}Br{combo["Br"]:.2f}Cl{combo["Cl"]:.2f})3'

        # 创建DataFrame
        new_sample = pd.DataFrame([new_sample_data])

        # 解析钙钛矿组成并添加元素比例 - 使用导入的解析函数
        element_ratios = parse_perovskite_composition(new_sample_data['Perovskite'])

        for element in ['Cs', 'MA', 'FA', 'I', 'Br', 'Cl']:
            new_sample[element] = element_ratios.get(element, 0.0)

        # 添加Bandgap特征
        element_cols = ['Cs', 'MA', 'FA', 'I', 'Br', 'Cl']
        new_sample['Bandgap'] = new_sample[element_cols].sum(axis=1)

        # 移除Perovskite列
        new_sample = new_sample.drop('Perovskite', axis=1)

        # 编码分类特征
        new_sample_encoded = encode_categorical_features(new_sample, mapping_df)

        # 确保所有列都是数值类型
        for col in new_sample_encoded.columns:
            if new_sample_encoded[col].dtype == 'object':
                try:
                    new_sample_encoded[col] = pd.to_numeric(new_sample_encoded[col])
                except:
                    new_sample_encoded[col] = 0

        # 调整特征顺序
        if "XGBoost" in models and hasattr(models["XGBoost"], 'feature_names_'):
            expected_features = models["XGBoost"].feature_names_
            # 添加缺失特征
            for feature in set(expected_features) - set(new_sample_encoded.columns):
                new_sample_encoded[feature] = 0
            # 移除多余特征
            new_sample_encoded = new_sample_encoded[expected_features]

        # 集成预测
        predictions = {}
        for name, model in models.items():
            try:
                prediction = model.predict(new_sample_encoded)[0]
                predictions[name] = prediction
            except Exception as e:
                predictions[name] = 0

        # 计算加权平均PCE
        if predictions:
            weighted_pce = sum(predictions[name] * weights[name] for name in predictions.keys())

            results.append({
                'I': combo['I'],
                'Br': combo['Br'],
                'Cl': combo['Cl'],
                'Cs': base_data['Cs'],
                'MA': base_data['MA'],
                'FA': base_data['FA'],
                'Bandgap': new_sample['Bandgap'].iloc[0],
                'Average_Predicted_PCE': weighted_pce,
                'RF_Prediction': predictions.get('RandomForest', 0),
                'XGB_Prediction': predictions.get('XGBoost', 0),
                'LGB_Prediction': predictions.get('LightGBM', 0)
            })

        # 显示进度
        if (i + 1) % 10 == 0:
            print(f"   已处理 {i + 1}/{len(combinations)} 个组合...")

    # 6. 分析结果
    if results:
        results_df = pd.DataFrame(results).sort_values('Average_Predicted_PCE', ascending=False)

        print(f"\n✅ 预测完成! 共生成 {len(results_df)} 个有效预测结果")

        # 显示前10个最佳结果
        print("\n🏆 预测结果排名前10的I和Br组合:")
        print("=" * 100)
        for i, row in results_df.head(10).iterrows():
            print(f"{i + 1:2d}. I: {row['I']:.2f}, Br: {row['Br']:.2f}, Cl: {row['Cl']:.2f}, "
                  f"Cs: {row['Cs']:.2f}, MA: {row['MA']:.2f}, FA: {row['FA']:.2f}, "
                  f"Bandgap: {row['Bandgap']:.3f}, Predicted_PCE: {row['Average_Predicted_PCE']:.2f}%")

        # 统计信息
        print(f"\n📊 预测结果统计:")
        print(f"   最高PCE: {results_df['Average_Predicted_PCE'].max():.2f}%")
        print(f"   最低PCE: {results_df['Average_Predicted_PCE'].min():.2f}%")
        print(f"   平均PCE: {results_df['Average_Predicted_PCE'].mean():.2f}%")
        print(f"   中位数PCE: {results_df['Average_Predicted_PCE'].median():.2f}%")

        # 最佳组合
        best_combo = results_df.iloc[0]
        print(f"\n⭐ 最佳组合推荐:")
        print(f"   I: {best_combo['I']:.2f}, Br: {best_combo['Br']:.2f}, Cl: {best_combo['Cl']:.2f}")
        print(f"   对应的Cs: {best_combo['Cs']:.2f}, MA: {best_combo['MA']:.2f}, FA: {best_combo['FA']:.2f}")
        print(f"   Bandgap: {best_combo['Bandgap']:.3f}")
        print(f"   预测PCE: {best_combo['Average_Predicted_PCE']:.2f}%")

        # 保存结果
        results_df.to_csv('pce_Predict/i_br_combinations_predictions.csv', index=False)
        print(f"\n💾 完整预测结果已保存到 pce_Predict/i_br_combinations_predictions.csv")

        # 保存前20个最佳结果
        results_df.head(20).to_csv('pce_Predict/i_br_best_combinations.csv', index=False)
        print(f"💾 前20个最佳结果已保存到 pce_Predict/i_br_best_combinations.csv")

        return results_df
    else:
        print("❌ 没有生成有效的预测结果")
        return None


if __name__ == "__main__":
    print("=== 钙钛矿太阳能电池I-Br组合PCE预测系统 ===\n")
    results = predict_i_br_combinations()