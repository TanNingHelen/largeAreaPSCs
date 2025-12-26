import pickle
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import warnings
from catboost import CatBoostRegressor
import joblib

warnings.filterwarnings('ignore')


def predict_bandgap(element_ratios):
    """
    使用训练好的CatBoost模型预测钙钛矿的Bandgap
    element_ratios: 包含元素比例的字典
    """
    try:
        # 加载Bandgap预测模型
        bandgap_model = CatBoostRegressor()
        bandgap_model.load_model('models/best_catboost_bandgap.cbm')
        print("✅ Bandgap模型加载成功")
    except Exception as e:
        print(f"❌ Bandgap模型加载失败: {e}")
        return None

    # 准备Bandgap预测特征
    bandgap_features = pd.DataFrame({
        'FA': [element_ratios['FA']],
        'MA': [element_ratios['MA']],
        'Cs': [element_ratios['Cs']],
        'I': [element_ratios['I']],
        'Br': [element_ratios['Br']],
        'Cl': [element_ratios['Cl']],
        'Pb': [element_ratios['Pb']]
    })

    print("🔬 使用元素比例预测Bandgap:")
    print(f"   FA: {element_ratios['FA']:.4f}, MA: {element_ratios['MA']:.4f}, Cs: {element_ratios['Cs']:.4f}")
    print(
        f"   I: {element_ratios['I']:.4f}, Br: {element_ratios['Br']:.4f}, Cl: {element_ratios['Cl']:.4f}, Pb: {element_ratios['Pb']:.4f}")

    # 预测Bandgap
    try:
        predicted_bandgap = bandgap_model.predict(bandgap_features)[0]
        print(f"   📊 预测Bandgap: {predicted_bandgap:.4f} eV")
        return predicted_bandgap
    except Exception as e:
        print(f"❌ Bandgap预测失败: {e}")
        return None


def prepare_sample_data(sample_data, mapping_df, historical_data, use_predicted_bandgap=True, predicted_bandgap=None):
    """
    准备样本数据并进行预处理

    Parameters:
    - sample_data: 样本数据字典
    - mapping_df: 映射数据框
    - historical_data: 历史数据
    - use_predicted_bandgap: 是否使用预测的带隙
    - predicted_bandgap: 预测的带隙值（如果use_predicted_bandgap为True且提供了值）
    """
    # 处理Bandgap
    if use_predicted_bandgap and predicted_bandgap is not None:
        # 使用预测的Bandgap值
        sample_data['Bandgap'] = predicted_bandgap
        print(f"✅ 使用预测Bandgap: {sample_data['Bandgap']:.4f} eV")
    elif use_predicted_bandgap:
        # 预测Bandgap（使用已有的元素比例）
        print("\n🔬 开始Bandgap预测...")
        element_ratios = {
            'FA': sample_data['FA'],
            'MA': sample_data['MA'],
            'Cs': sample_data['Cs'],
            'I': sample_data['I'],
            'Br': sample_data['Br'],
            'Cl': sample_data['Cl'],
            'Pb': sample_data['Pb']
        }

        predicted_bandgap = predict_bandgap(element_ratios)

        if predicted_bandgap is not None:
            # 添加预测的Bandgap到特征中
            sample_data['Bandgap'] = predicted_bandgap
            print(f"✅ 已添加预测Bandgap: {predicted_bandgap:.4f} eV")
        else:
            # 如果Bandgap预测失败，使用元素比例之和作为替代
            element_cols = ['Cs', 'MA', 'FA', 'I', 'Br', 'Cl', 'Pb']
            sample_data['Bandgap'] = sum(sample_data[col] for col in element_cols)
            print(f"⚠️  Bandgap预测失败，使用元素比例之和: {sample_data['Bandgap']:.4f}")
    else:
        # 使用给定的Bandgap值
        sample_data['Bandgap'] = 1.6039
        print(f"✅ 使用给定Bandgap: {sample_data['Bandgap']:.4f} eV")

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


def predict_pce_for_new_samples():
    """
    使用训练好的CatBoost模型预测新实验数据的PCE
    """
    # 1. 加载PCE预测模型
    try:
        # 使用CatBoost的load_model方法加载.cbm文件
        model = CatBoostRegressor()
        model.load_model('models/best_catboost_model.cbm')
        print("✅ CatBoost PCE模型加载成功")

        # 打印模型信息
        print(f"📋 模型特征数量: {model.feature_count_ if hasattr(model, 'feature_count_') else '未知'}")

    except Exception as e:
        print(f"❌ CatBoost PCE模型加载失败: {e}")
        return None

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
        'Perovskite': '(FA0.98MA0.02)0.95Cs0.05Pb(I0.98Br0.02)3',
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
        'Cl': 0
    }

    # 5. 准备第二组数据（HTL改为Me-4PACz，HTL-2改为空值）
    sample2_data = sample1_data.copy()  # 复制第一组数据
    sample2_data['HTL'] = 'Me-4PACz'  # 修改HTL
    sample2_data['HTL-2'] = ''  # HTL-2改为空值
    sample2_data['HTL-Addictive'] = ''  # HTL-Addictive改为空值
    sample2_data['Glass'] = 'ITO'

    # 6. 准备第三组数据（HTL-2和HTL-Addictive都变为空值）
    sample3_data = sample1_data.copy()  # 复制第一组数据
    sample3_data['HTL-2'] = ''  # HTL-2改为空值
    sample3_data['HTL-Addictive'] = ''  # HTL-Addictive改为空值
    sample3_data['Glass'] = 'ITO'

    # 7. 准备第四组数据（HTL-Addictive换成空值）
    sample4_data = sample1_data.copy()  # 复制第一组数据
    sample4_data['HTL-Addictive'] = ''  # HTL-Addictive改为空值
    sample4_data['Glass'] = 'ITO'

    # 8. 准备第五组数据（HTL-Addictive换成空值，同时ETL_Passivator变成LiF）
    sample5_data = sample1_data.copy()  # 复制第一组数据
    sample5_data['HTL-Addictive'] = ''  # HTL-Addictive改为空值
    sample5_data['ETL_Passivator'] = 'LiF'  # ETL_Passivator改为LiF
    sample5_data['Glass'] = 'ITO'

    # 9. 准备第六组数据（HTL-Addictive变成DMPU+PEAI，同时ETL_Passivator变成LiF）
    sample6_data = sample1_data.copy()  # 复制第一组数据
    sample6_data['HTL-Addictive'] = 'DMPU+PEAI'  # HTL-Addictive改为DMPU+PEAI
    sample6_data['ETL_Passivator'] = 'LiF'  # ETL_Passivator改为LiF
    sample6_data['Glass'] = 'ITO'

    # 10. 准备第七组数据（新增实验配置）
    sample7_data = sample1_data.copy()  # 复制第一组数据
    # 修改指定的参数
    sample7_data['HTL-Addictive'] = 'DMPU+PEAI'  # HTL-Addictive改为DMPU+PEAI
    sample7_data['ETL_Passivator'] = 'LiF'  # ETL_Passivator改为LiF
    sample7_data['Glass'] = 'ITO'  # Glass改成ITO
    sample7_data['Precursor_Solution_Addictive'] = 'PbCl2+FAI'  # Precursor_Solution_Addictive改成PbCl2+FAI

    # 11. 准备第八组数据（新增实验配置）
    sample8_data = sample1_data.copy()  # 复制第一组数据
    # 修改指定的参数
    sample8_data['HTL-Addictive'] = 'DMPU+PEAI'  # HTL-Addictive改为DMPU+PEAI
    sample8_data['Precursor_Solution_Addictive'] = 'PbCl2+FAI'  # Precursor_Solution_Addictive改成PbCl2+FAI
    sample8_data['ETL_Passivator'] = 'PDAI'  # ETL_Passivator改为PDAI
    sample8_data['Glass'] = 'ITO'  # Glass改成ITO

    # 存储所有预测结果
    all_results = {}

    # 存储第一组数据预测的带隙值
    first_sample_bandgap = None

    print("=" * 60)
    print("🎯 第一组数据预测 (基准配置)")
    print("=" * 60)

    # 准备第一组数据（进行带隙预测）
    sample1_processed = prepare_sample_data(sample1_data, mapping_df, historical_data, use_predicted_bandgap=True)
    first_sample_bandgap = sample1_data.get('Bandgap', None)

    # 预测第一组数据的PCE
    try:
        pce_prediction1 = model.predict(sample1_processed)[0]
        print(f"\n🎯 第一组数据预测结果:")
        print(f"   预测PCE: {pce_prediction1:.2f} %")
        if first_sample_bandgap is not None:
            print(f"   预测Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample1'] = {'pce': pce_prediction1, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第一组数据PCE预测失败: {e}")
        all_results['sample1'] = {'pce': None, 'bandgap': first_sample_bandgap}

    print("\n" + "=" * 60)
    print("🎯 第二组数据预测 (HTL: Me-4PACz, HTL-2: 空值)")
    print("=" * 60)
    print(f"使用第一组数据预测的Bandgap: {first_sample_bandgap:.4f} eV")

    # 准备第二组数据（使用第一组数据预测的带隙）
    sample2_processed = prepare_sample_data(sample2_data, mapping_df, historical_data,
                                            use_predicted_bandgap=True, predicted_bandgap=first_sample_bandgap)

    # 预测第二组数据的PCE
    try:
        pce_prediction2 = model.predict(sample2_processed)[0]
        print(f"\n🎯 第二组数据预测结果:")
        print(f"   预测PCE: {pce_prediction2:.2f} %")
        print(f"   使用Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample2'] = {'pce': pce_prediction2, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第二组数据PCE预测失败: {e}")
        all_results['sample2'] = {'pce': None, 'bandgap': first_sample_bandgap}

    print("\n" + "=" * 60)
    print("🎯 第三组数据预测 (HTL-2: 空值, HTL-Addictive: 空值)")
    print("=" * 60)
    print(f"使用第一组数据预测的Bandgap: {first_sample_bandgap:.4f} eV")

    # 准备第三组数据（使用第一组数据预测的带隙）
    sample3_processed = prepare_sample_data(sample3_data, mapping_df, historical_data,
                                            use_predicted_bandgap=True, predicted_bandgap=first_sample_bandgap)

    # 预测第三组数据的PCE
    try:
        pce_prediction3 = model.predict(sample3_processed)[0]
        print(f"\n🎯 第三组数据预测结果:")
        print(f"   预测PCE: {pce_prediction3:.2f} %")
        print(f"   使用Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample3'] = {'pce': pce_prediction3, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第三组数据PCE预测失败: {e}")
        all_results['sample3'] = {'pce': None, 'bandgap': first_sample_bandgap}

    print("\n" + "=" * 60)
    print("🎯 第四组数据预测 (HTL-Addictive: 空值)")
    print("=" * 60)
    print(f"使用第一组数据预测的Bandgap: {first_sample_bandgap:.4f} eV")

    # 准备第四组数据（使用第一组数据预测的带隙）
    sample4_processed = prepare_sample_data(sample4_data, mapping_df, historical_data,
                                            use_predicted_bandgap=True, predicted_bandgap=first_sample_bandgap)

    # 预测第四组数据的PCE
    try:
        pce_prediction4 = model.predict(sample4_processed)[0]
        print(f"\n🎯 第四组数据预测结果:")
        print(f"   预测PCE: {pce_prediction4:.2f} %")
        print(f"   使用Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample4'] = {'pce': pce_prediction4, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第四组数据PCE预测失败: {e}")
        all_results['sample4'] = {'pce': None, 'bandgap': first_sample_bandgap}

    print("\n" + "=" * 60)
    print("🎯 第五组数据预测 (HTL-Addictive: 空值, ETL_Passivator: LiF)")
    print("=" * 60)
    print(f"使用第一组数据预测的Bandgap: {first_sample_bandgap:.4f} eV")

    # 准备第五组数据（使用第一组数据预测的带隙）
    sample5_processed = prepare_sample_data(sample5_data, mapping_df, historical_data,
                                            use_predicted_bandgap=True, predicted_bandgap=first_sample_bandgap)

    # 预测第五组数据的PCE
    try:
        pce_prediction5 = model.predict(sample5_processed)[0]
        print(f"\n🎯 第五组数据预测结果:")
        print(f"   预测PCE: {pce_prediction5:.2f} %")
        print(f"   使用Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample5'] = {'pce': pce_prediction5, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第五组数据PCE预测失败: {e}")
        all_results['sample5'] = {'pce': None, 'bandgap': first_sample_bandgap}

    print("\n" + "=" * 60)
    print("🎯 第六组数据预测 (HTL-Addictive: DMPU+PEAI, ETL_Passivator: LiF)")
    print("=" * 60)
    print(f"使用第一组数据预测的Bandgap: {first_sample_bandgap:.4f} eV")

    # 准备第六组数据（使用第一组数据预测的带隙）
    sample6_processed = prepare_sample_data(sample6_data, mapping_df, historical_data,
                                            use_predicted_bandgap=True, predicted_bandgap=first_sample_bandgap)

    # 预测第六组数据的PCE
    try:
        pce_prediction6 = model.predict(sample6_processed)[0]
        print(f"\n🎯 第六组数据预测结果:")
        print(f"   预测PCE: {pce_prediction6:.2f} %")
        print(f"   使用Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample6'] = {'pce': pce_prediction6, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第六组数据PCE预测失败: {e}")
        all_results['sample6'] = {'pce': None, 'bandgap': first_sample_bandgap}

    print("\n" + "=" * 60)
    print("🎯 第七组数据预测 (新增实验配置)")
    print("=" * 60)
    print(f"使用第一组数据预测的Bandgap: {first_sample_bandgap:.4f} eV")

    # 准备第七组数据（使用第一组数据预测的带隙）
    sample7_processed = prepare_sample_data(sample7_data, mapping_df, historical_data,
                                            use_predicted_bandgap=True, predicted_bandgap=first_sample_bandgap)

    # 预测第七组数据的PCE
    try:
        pce_prediction7 = model.predict(sample7_processed)[0]
        print(f"\n🎯 第七组数据预测结果:")
        print(f"   预测PCE: {pce_prediction7:.2f} %")
        print(f"   使用Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample7'] = {'pce': pce_prediction7, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第七组数据PCE预测失败: {e}")
        all_results['sample7'] = {'pce': None, 'bandgap': first_sample_bandgap}

    print("\n" + "=" * 60)
    print("🎯 第八组数据预测 (新增实验配置)")
    print("=" * 60)
    print(f"使用第一组数据预测的Bandgap: {first_sample_bandgap:.4f} eV")

    # 准备第八组数据（使用第一组数据预测的带隙）
    sample8_processed = prepare_sample_data(sample8_data, mapping_df, historical_data,
                                            use_predicted_bandgap=True, predicted_bandgap=first_sample_bandgap)

    # 预测第八组数据的PCE
    try:
        pce_prediction8 = model.predict(sample8_processed)[0]
        print(f"\n🎯 第八组数据预测结果:")
        print(f"   预测PCE: {pce_prediction8:.2f} %")
        print(f"   使用Bandgap: {first_sample_bandgap:.4f} eV")
        all_results['sample8'] = {'pce': pce_prediction8, 'bandgap': first_sample_bandgap}
    except Exception as e:
        print(f"❌ 第八组数据PCE预测失败: {e}")
        all_results['sample8'] = {'pce': None, 'bandgap': first_sample_bandgap}

    return all_results


# 主函数
if __name__ == "__main__":
    print("=== 钙钛矿太阳能电池PCE预测系统 (CatBoost) ===\n")
    print("本系统将预测八种不同配置的PCE性能")
    print("配置1 (基准): HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU, ETL_Passivator = 空值")
    print("配置2: HTL = Me-4PACz, HTL-2 = 空值, HTL-Addictive = DMPU, ETL_Passivator = 空值")
    print("配置3: HTL = NiOx, HTL-2 = 空值, HTL-Addictive = 空值, ETL_Passivator = 空值")
    print("配置4: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = 空值, ETL_Passivator = 空值")
    print("配置5: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = 空值, ETL_Passivator = LiF")
    print("配置6: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU+PEAI, ETL_Passivator = LiF")
    print("配置7: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU+PEAI, ETL_Passivator = LiF, Glass = ITO, Precursor_Solution_Addictive = PbCl2+FAI")
    print("配置8: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU+PEAI, ETL_Passivator = PDAI, Glass = ITO, Precursor_Solution_Addictive = PbCl2+FAI")
    print("所有配置使用相同的元素比例，仅第一组数据计算Bandgap，后续组使用第一组的Bandgap预测值")
    print("其他参数保持不变\n")

    # 预测八组数据的PCE
    results = predict_pce_for_new_samples()

    if results:
        print("\n" + "=" * 60)
        print("📊 所有预测结果汇总")
        print("=" * 60)

        config_names = {
            'sample1': "HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU, ETL_Passivator = 空值",
            'sample2': "HTL = Me-4PACz, HTL-2 = 空值, HTL-Addictive = 空值, ETL_Passivator = 空值",
            'sample3': "HTL = NiOx, HTL-2 = 空值, HTL-Addictive = 空值, ETL_Passivator = 空值",
            'sample4': "HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = 空值, ETL_Passivator = 空值",
            'sample5': "HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = 空值, ETL_Passivator = LiF",
            'sample6': "HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU+PEAI, ETL_Passivator = LiF",
            'sample7': "HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU+PEAI, ETL_Passivator = LiF, Glass = ITO, Precursor_Solution_Addictive = PbCl2+FAI",
            'sample8': "HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU+PEAI, ETL_Passivator = PDAI, Glass = ITO, Precursor_Solution_Addictive = PbCl2+FAI"
        }

        for i, (sample_key, result) in enumerate(results.items(), 1):
            if result['pce'] is not None:
                config_name = config_names.get(sample_key, f"配置{i}")
                print(f"配置{i}:")
                print(f"  {config_name}")
                print(f"  PCE = {result['pce']:.2f} %")
                print(f"  Bandgap = {result['bandgap']:.4f} eV")

                # 提供性能评估
                if result['pce'] > 20:
                    print("  ⭐ 优秀性能!")
                elif result['pce'] > 18:
                    print("  👍 良好性能!")
                else:
                    print("  💡 建议进一步优化工艺参数!")
            else:
                print(f"配置{i}: 预测失败")

            print()

        # Bandgap参考信息
        if results['sample1']['bandgap'] is not None:
            print(f"🔬 Bandgap信息: {results['sample1']['bandgap']:.4f} eV")
            if results['sample1']['bandgap'] < 1.5:
                print("   💡 Bandgap较低，可能适合串联电池应用")
            elif results['sample1']['bandgap'] > 1.7:
                print("   💡 Bandgap较高，可能获得较高开路电压")
            else:
                print("   💡 Bandgap适中，适合单结电池应用")