import pickle
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


def prepare_sample_data(sample_data, mapping_df, historical_data, fixed_bandgap=1.6095):
    """
    准备样本数据并进行预处理

    Parameters:
    - sample_data: 样本数据字典
    - mapping_df: 映射数据框
    - historical_data: 历史数据
    - fixed_bandgap: 固定的Bandgap值
    """
    # 使用固定的Bandgap值
    sample_data['Bandgap'] = fixed_bandgap
    print(f"✅ 使用固定Bandgap: {sample_data['Bandgap']:.4f} eV")

    # 创建新数据的DataFrame
    new_sample = pd.DataFrame([sample_data])

    # 移除Perovskite列（因为已经有元素比例和Bandgap）
    if 'Perovskite' in new_sample.columns:
        new_sample = new_sample.drop('Perovskite', axis=1)
        print("✅ 已移除Perovskite列，保留元素比例和Bandgap特征")

    # 应用数值映射
    categorical_features = [
        'Structure', 'HTL', 'HTL-2', 'HTL_Passivator', 'HTL-Addictive',
        'ETL', 'ETL-2', 'ETL_Passivator', 'ETL-Addictive',
        'Metal_Electrode', 'Glass', 'Precursor_Solution',
        'Precursor_Solution_Addictive', 'Deposition_Method',
        'Antisolvent', 'Type', 'brand'
    ]

    print("\n🔧 开始特征编码...")

    for feature in categorical_features:
        if feature in new_sample.columns:
            # 获取该特征的映射关系
            feature_mapping = mapping_df[mapping_df['Feature'] == feature]

            if len(feature_mapping) > 0:
                # 创建映射字典
                mapping_dict = dict(zip(feature_mapping['Original'], feature_mapping['Encoded']))

                # 应用映射
                original_value = new_sample[feature].iloc[0]

                # 处理空值
                if original_value == '' or pd.isna(original_value):
                    # 查找空值的映射
                    empty_mapping = feature_mapping[feature_mapping['Original'].isna()]
                    if len(empty_mapping) > 0:
                        encoded_value = empty_mapping['Encoded'].iloc[0]
                    else:
                        # 如果没有空值映射，使用0
                        encoded_value = 0
                else:
                    # 正常映射
                    encoded_value = mapping_dict.get(original_value, 0)

                new_sample[feature] = encoded_value
                print(f"   {feature}: '{original_value}' -> {encoded_value}")
            else:
                print(f"   ⚠️  特征 '{feature}' 在映射文件中未找到，使用默认值0")
                new_sample[feature] = 0

    # 确保所有列都是数值类型
    for col in new_sample.columns:
        if new_sample[col].dtype == 'object':
            try:
                new_sample[col] = pd.to_numeric(new_sample[col])
            except:
                print(f"   ⚠️  无法将列 '{col}' 转换为数值类型，使用0")
                new_sample[col] = 0

    # 确保特征顺序与训练时一致
    try:
        # 获取历史数据的特征顺序（排除目标变量PCE）
        expected_features = [col for col in historical_data.columns if col != 'PCE']

        print(f"\n📋 期望的特征数量: {len(expected_features)}")

        # 检查缺失和多余的特征
        missing_features = set(expected_features) - set(new_sample.columns)
        extra_features = set(new_sample.columns) - set(expected_features)

        print(f"🔍 特征匹配检查:")
        print(f"   缺失特征: {missing_features}")
        print(f"   多余特征: {extra_features}")

        # 添加缺失特征
        for feature in missing_features:
            print(f"   ➕ 添加缺失特征: {feature} = 0")
            new_sample[feature] = 0

        # 移除多余特征
        if extra_features:
            print(f"   ➖ 移除多余特征: {extra_features}")
            new_sample = new_sample.drop(columns=list(extra_features))

        # 重新排列列顺序
        new_sample = new_sample[expected_features]
        print(f"   ✅ 特征顺序已调整，当前特征数量: {len(new_sample.columns)}")

    except Exception as e:
        print(f"⚠️  特征顺序调整失败: {e}")

    return new_sample


def predict_pce_for_first_sample():
    """
    使用三个不同的模型分别预测第一组原始数据的PCE
    """
    # 1. 加载三个PCE预测模型
    models = {}
    try:
        # 加载随机森林模型
        rf_model = joblib.load('models/best_randomforest_model.pkl')
        models['Random Forest'] = rf_model
        print("✅ 随机森林模型加载成功")
    except Exception as e:
        print(f"❌ 随机森林模型加载失败: {e}")
        return None

    try:
        # 加载LightGBM模型
        lgb_model = joblib.load('models/best_lgbm_model.pkl')
        models['LightGBM'] = lgb_model
        print("✅ LightGBM模型加载成功")
    except Exception as e:
        print(f"❌ LightGBM模型加载失败: {e}")
        return None

    try:
        # 加载XGBoost模型
        xgb_model = joblib.load('models/best_xgboost_model.pkl')
        models['XGBoost'] = xgb_model
        print("✅ XGBoost模型加载成功")
    except Exception as e:
        print(f"❌ XGBoost模型加载失败: {e}")
        return None

    print(f"📋 加载了 {len(models)} 个模型")

    # 2. 加载历史数据以获取特征结构
    try:
        historical_data = pd.read_excel('FinalData.xlsx')
        print("✅ 历史数据加载成功")
        print(f"历史数据特征数量: {len(historical_data.columns)}")
    except Exception as e:
        print(f"❌ 历史数据加载失败: {e}")
        return None

    # 3. 加载映射文件
    try:
        mapping_df = pd.read_csv('label_mappings/full_mapping_summary.csv')
        print("✅ 映射文件加载成功")
    except Exception as e:
        print(f"❌ 映射文件加载失败: {e}")
        return None

    # 4. 准备第一组数据（原始数据）
    sample1_data = {
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
        'Perovskite': '(FA0.98MA0.02)0.95Cs0.05Pb(l0.98Br0.02)3',
        'Active_Area': 12.96,
        'Precursor_Solution': 'DMF:NMP (7:1)',
        'Precursor_Solution_Addictive': '',
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
        'I': 2.94,
        'Br': 0.96,
        'Pb': 1.0,
        'Cl': 0,
        'Bandgap': 1.6095  # 固定Bandgap值
    }

    # 存储所有预测结果
    all_predictions = {}

    print("=" * 60)
    print("🎯 第一组数据预测 (原始配置)")
    print("=" * 60)
    print("配置: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU, ETL_Passivator = 空值")
    print(f"使用固定Bandgap值: 1.6095 eV")

    # 准备第一组数据
    sample1_processed = prepare_sample_data(sample1_data, mapping_df, historical_data, fixed_bandgap=1.6095)

    # 分别用三个模型进行预测
    for model_name, model in models.items():
        try:
            pce_prediction = model.predict(sample1_processed)[0]
            all_predictions[model_name] = pce_prediction
            print(f"\n📊 {model_name} 预测结果:")
            print(f"   预测PCE: {pce_prediction:.2f} %")

            # 提供性能评估
            if pce_prediction > 20:
                print("   ⭐ 优秀性能!")
            elif pce_prediction > 18:
                print("   👍 良好性能!")
            else:
                print("   💡 建议进一步优化工艺参数!")
        except Exception as e:
            print(f"\n❌ {model_name} 预测失败: {e}")
            all_predictions[model_name] = None

    return all_predictions


# 主函数
if __name__ == "__main__":
    print("=== 钙钛矿太阳能电池PCE预测系统 ===")
    print("使用三个模型分别预测第一组原始数据")
    print("配置: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU, ETL_Passivator = 空值")
    print("使用固定Bandgap值: 1.6095 eV")
    print("预测模型: Random Forest, LightGBM, XGBoost")
    print("=" * 60)

    # 预测第一组数据的PCE
    predictions = predict_pce_for_first_sample()

    if predictions:
        print("\n" + "=" * 60)
        print("📊 所有模型预测结果汇总")
        print("=" * 60)

        for model_name, pce in predictions.items():
            if pce is not None:
                print(f"{model_name}: {pce:.2f} %")
            else:
                print(f"{model_name}: 预测失败")

        print("\n" + "=" * 60)
        print("📈 预测结果统计")
        print("=" * 60)

        # 计算统计信息
        valid_predictions = [p for p in predictions.values() if p is not None]
        if valid_predictions:
            print(f"预测模型数量: {len(valid_predictions)}")
            print(f"平均预测PCE: {np.mean(valid_predictions):.2f} %")
            print(f"最高预测PCE: {max(valid_predictions):.2f} %")
            print(f"最低预测PCE: {min(valid_predictions):.2f} %")
            print(f"预测PCE范围: {max(valid_predictions) - min(valid_predictions):.2f} %")
        else:
            print("所有模型预测都失败了")
    else:
        print("预测失败，请检查模型和数据")