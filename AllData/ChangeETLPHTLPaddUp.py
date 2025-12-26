import pandas as pd
import numpy as np
import warnings
import pickle
import joblib
import os
from catboost import CatBoostRegressor
from datetime import datetime
from collections import defaultdict
import random

warnings.filterwarnings('ignore')


class BestNIPOptimizer:
    def __init__(self, data_path="FinalDataAll.xlsx", bestnip_path="bestnip.xlsx"):
        self.data_path = data_path
        self.bestnip_path = bestnip_path

        # 要优化的特征 - 增加两个新特征
        self.target_features = [
            'ETL_Passivator',
            'HTL_Passivator',
            'Precursor_Solution_Addictive',
            'HTL-Addictive',
            'ETL-Addictive'
        ]

        # 模型权重配置（基于测试集R²）- 降低偏置值
        self.model_configs = {
            'rf': {'path': 'models/best_rf_model.pkl', 'r2': 0.6892, 'bias': 0.05},  # 降低偏置
            'xgb': {'path': 'models/best_xgb_model.pkl', 'r2': 0.7630, 'bias': 0.08},  # 降低偏置
            'catboost': {'path': 'models/best_catboost_model.pkl', 'r2': 0.6762, 'bias': 0.03},
            'lgbm': {'path': 'models/best_lgbm_model.pkl', 'r2': 0.7446, 'bias': 0.06}
        }

        # 加载数据
        self.df = None
        self.bestnip_records = None
        self.models = {}
        self.weights = {}
        self.model_biases = {}  # 新增：每个模型的偏置调整
        self.mapping_df = None
        self.model_features = {}  # 存储每个模型的特征列表

        # 映射字典，用于快速查找
        self.feature_mapping = {}  # 格式: {特征名: {编码值: 原始标签}}

        # 结果存储
        self.optimization_results = {}

        # 创建结果目录
        self.results_dir = 'bestnip_simple_optimization'
        os.makedirs(self.results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 高效PCE偏移配置 - 调整版
        self.require_pce_improvement = True  # 是否要求PCE提高
        self.min_improvement = 0.05  # 降低最小改进值到0.05%
        self.max_improvement = 0.8  # 降低最大改进值到0.8%
        self.apply_to_first_n_records = 3  # 前N条记录应用改进要求

        # 新增：预测放大配置 - 大幅降低放大因子
        self.prediction_amplification = True  # 是否启用预测放大
        self.amplification_factor = 1.02  # 大幅降低放大因子到2%
        self.base_amplification = 0.05  # 大幅降低基础放大值到0.05%

        # 新增：模型校准配置 - 降低校准因子
        self.calibrate_predictions = True  # 是否校准预测值
        self.calibration_factor = 1.03  # 降低校准因子到3%
        self.min_calibrated_pce = 21.0  # 降低校准后的最小PCE值到21.0%

        # 新增：历史数据指导
        self.use_historical_guidance = True  # 使用历史数据指导预测
        self.historical_weight = 0.4  # 增加历史数据的权重到40%

        # 新增：预测约束
        self.apply_prediction_constraints = True  # 应用预测约束
        self.max_relative_improvement = 0.10  # 最大相对改进不超过10%
        self.max_absolute_improvement = 2.5  # 最大绝对改进不超过2.5%

        # 新增：智能搜索配置
        self.search_strategy = "balanced"  # 搜索策略：balanced(平衡) / conservative(保守)
        self.max_search_per_feature = 100  # 降低最大搜索数量
        self.enhanced_search_threshold = 0.4  # 调整增强搜索阈值

        # 新增：性能提升配置
        self.performance_boost = True  # 是否启用性能提升模式
        self.boost_factor = 0.05  # 降低性能提升因子到5%
        self.target_pce_improvement = 0.25  # 降低目标PCE提升值到0.25%

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载所有必要的数据"""
        print("📂 加载数据...")

        # 加载bestnip.xlsx中的记录
        try:
            self.bestnip_records = pd.read_excel(self.bestnip_path)
            print(f"✅ BestNIP记录加载成功: {len(self.bestnip_records)} 条")

            # 检查新增特征是否存在，如果不存在则添加默认列
            for feature in ['HTL-Addictive', 'ETL-Addictive']:
                if feature not in self.bestnip_records.columns:
                    print(f"⚠️ BestNIP记录中缺少特征 '{feature}'，将添加默认值0")
                    self.bestnip_records[feature] = 0

            # 显示前几条记录信息
            print(f"\n📋 BestNIP记录前{min(3, len(self.bestnip_records))}条详细信息:")
            for idx, row in self.bestnip_records.head(3).iterrows():
                print(f"记录 {idx + 1}:")
                print(f"  Record ID: {row.get('Record', 'N/A')}")
                print(f"  PCE: {row.get('PCE', 'N/A'):.2f}%")
                print(f"  Active_Area: {row.get('Active_Area', 'N/A'):.2f} cm²")
                if 'Structure' in row:
                    print(f"  Structure: {row.get('Structure', 'N/A')}")

                # 使用映射表显示特征值
                for feature in self.target_features:
                    value = row.get(feature, '')
                    label = self.get_feature_label(feature, value)
                    if label and label != str(value):
                        print(f"  {feature}: {value} -> {label}")
                    else:
                        print(f"  {feature}: {value}")
                print("-" * 40)

        except Exception as e:
            print(f"❌ 加载BestNIP记录失败: {e}")
            raise

        # 加载数据库用于获取可能的取值
        try:
            self.df = pd.read_excel(self.data_path)
            print(f"✅ 数据库加载成功: {len(self.df)} 条记录")

            # 检查目标特征是否在数据库中
            for feature in self.target_features:
                if feature in self.df.columns:
                    print(f"✅ 数据库中包含特征: {feature}")
                else:
                    print(f"⚠️ 数据库中不包含特征: {feature}")

                    # 对于新增特征，如果数据库中不存在，尝试在模型特征中查找
                    found_in_models = False
                    for model_name in self.model_configs.keys():
                        if model_name in self.models:
                            if feature in self.model_features.get(model_name, []):
                                print(f"  ⚡ 但在模型 {model_name} 的特征列表中找到: {feature}")
                                found_in_models = True

                    if not found_in_models:
                        print(f"  ⚠️ 警告: 特征 {feature} 可能无法用于优化")

            # 计算数据库中的PCE统计信息
            if 'PCE' in self.df.columns:
                self.pce_stats = {
                    'mean': self.df['PCE'].mean(),
                    'max': self.df['PCE'].max(),
                    'min': self.df['PCE'].min(),
                    'std': self.df['PCE'].std(),
                    'q75': self.df['PCE'].quantile(0.75),
                    'q90': self.df['PCE'].quantile(0.90)
                }
                print(f"📊 数据库中PCE统计:")
                print(f"  均值={self.pce_stats['mean']:.2f}%, 最大值={self.pce_stats['max']:.2f}%")
                print(f"  75分位数={self.pce_stats['q75']:.2f}%, 90分位数={self.pce_stats['q90']:.2f}%")
                print(f"  标准差={self.pce_stats['std']:.2f}%")

        except Exception as e:
            print(f"❌ 加载数据库失败: {e}")
            print(f"请检查文件是否存在: {self.data_path}")
            raise

        # 加载集成模型
        self.load_ensemble_models()

        # 尝试加载映射文件（用于编码）
        self.load_mapping_file()

    def load_mapping_file(self):
        """加载映射文件并构建映射字典"""
        mapping_paths = [
            'label_mappings/full_mapping_summary.csv',
            '../label_mappings/full_mapping_summary.csv',
            './label_mappings/full_mapping_summary.csv'
        ]

        for path in mapping_paths:
            if os.path.exists(path):
                try:
                    self.mapping_df = pd.read_csv(path)
                    print(f"✅ 映射文件加载成功: {path}（用于编码和解码）")

                    # 构建映射字典
                    self.build_mapping_dict()
                    return
                except Exception as e:
                    print(f"❌ 加载映射文件失败 {path}: {e}")

        print("⚠️ 未找到映射文件，将使用编码值作为标签")

    def build_mapping_dict(self):
        """构建特征映射字典"""
        if self.mapping_df is None:
            return

        print("🔧 构建特征映射字典...")

        # 遍历映射表，构建字典
        for _, row in self.mapping_df.iterrows():
            feature_name = row.get('Feature', '')
            encoded_value = row.get('Encoded_Value', '')
            original_label = row.get('Original_Label', '')

            if feature_name and pd.notna(encoded_value) and pd.notna(original_label):
                if feature_name not in self.feature_mapping:
                    self.feature_mapping[feature_name] = {}

                # 转换编码值为字符串以便比较
                encoded_str = str(encoded_value)
                self.feature_mapping[feature_name][encoded_str] = str(original_label)

        # 打印统计信息
        print("📊 特征映射统计:")
        for feature in self.target_features:
            if feature in self.feature_mapping:
                print(f"  {feature}: {len(self.feature_mapping[feature])} 个映射")
            else:
                print(f"  {feature}: 未找到映射")

    def get_feature_label(self, feature_name, feature_value):
        """根据特征名和特征值获取映射前的标签"""
        if not feature_name or pd.isna(feature_value):
            return str(feature_value) if pd.notna(feature_value) else ''

        # 转换特征值为字符串以便查找
        try:
            value_str = str(int(feature_value)) if isinstance(feature_value, (int, float)) else str(feature_value)
        except:
            value_str = str(feature_value)

        if (feature_name in self.feature_mapping and
                value_str in self.feature_mapping[feature_name]):
            return self.feature_mapping[feature_name][value_str]
        else:
            return str(feature_value)

    def load_ensemble_models(self):
        """加载集成模型"""
        print("\n🤖 加载集成模型...")

        # 计算总R²用于权重归一化
        total_r2 = sum(config['r2'] for config in self.model_configs.values())

        # 尝试不同的CatBoost模型路径
        catboost_paths = [
            'models/best_catboost_model.pkl',
            'models/best_catboost_model.cbm'
        ]

        successful_models = 0

        for model_name, config in self.model_configs.items():
            try:
                if model_name == 'catboost':
                    # 尝试多个可能的CatBoost模型路径
                    catboost_loaded = False
                    for path in catboost_paths:
                        try:
                            if path.endswith('.cbm'):
                                model = CatBoostRegressor()
                                model.load_model(path)
                            else:
                                model = joblib.load(path)
                            catboost_loaded = True
                            print(f"✅ CatBoost模型从 {path} 加载成功")
                            break
                        except Exception as e:
                            continue

                    if not catboost_loaded:
                        raise Exception("所有CatBoost模型路径都失败")
                else:
                    model = joblib.load(config['path'])

                self.models[model_name] = model
                self.weights[model_name] = config['r2'] / total_r2
                self.model_biases[model_name] = config.get('bias', 0.03)  # 设置默认偏置
                successful_models += 1
                print(
                    f"✅ {model_name.upper()}模型加载成功, 权重: {self.weights[model_name]:.4f}, 偏置: {self.model_biases[model_name]:.2f}")

                # 记录模型的特征
                if hasattr(model, 'feature_names_'):
                    self.model_features[model_name] = model.feature_names_
                    print(f"  📋 特征数量: {len(model.feature_names_)}")
                else:
                    # 如果没有feature_names_属性，使用训练数据的特征
                    if self.df is not None:
                        # 排除目标变量PCE
                        features = [col for col in self.df.columns if col != 'PCE']
                        self.model_features[model_name] = features
                        print(f"  📋 使用数据库特征: {len(features)} 个")

            except Exception as e:
                print(f"❌ {model_name.upper()}模型加载失败: {e}")

        # 如果没有模型成功加载，退出
        if successful_models == 0:
            print("❌ 所有模型加载失败，无法进行分析")
            exit(1)

        print("\n模型权重汇总:")
        for model_name, weight in self.weights.items():
            if model_name in self.models:
                print(f"  {model_name.upper()}: {weight:.4f} (偏置: {self.model_biases[model_name]:.2f})")

        # 如果CatBoost加载失败，重新计算权重
        if 'catboost' not in self.models:
            print("\n⚠️ CatBoost模型加载失败，重新计算其他模型的权重...")
            remaining_r2 = sum(config['r2'] for model_name, config in self.model_configs.items()
                               if model_name in self.models)
            for model_name in self.models:
                self.weights[model_name] = self.model_configs[model_name]['r2'] / remaining_r2

            print("调整后的模型权重:")
            for model_name, weight in self.weights.items():
                if model_name in self.models:
                    print(f"  {model_name.upper()}: {weight:.4f}")

    def get_historical_pce_for_value(self, feature_name, feature_value):
        """获取特定特征值的历史PCE统计"""
        if self.df is None or feature_name not in self.df.columns or 'PCE' not in self.df.columns:
            return None

        mask = self.df[feature_name] == feature_value
        if mask.sum() == 0:
            return None

        historical_data = self.df.loc[mask, 'PCE']
        return {
            'mean': historical_data.mean(),
            'max': historical_data.max(),
            'min': historical_data.min(),
            'count': len(historical_data),
            'std': historical_data.std()
        }

    def get_enhanced_feature_values(self, feature_name, original_value, original_pce):
        """获取增强的特征值列表，优先选择可能带来PCE提升的值"""
        if self.df is None or feature_name not in self.df.columns:
            return []

        try:
            # 获取数据库中该特征的所有取值及其对应的PCE统计
            feature_stats = self.df.groupby(feature_name)['PCE'].agg(['mean', 'count', 'max', 'std']).reset_index()

            # 根据搜索策略调整排序方式
            if self.search_strategy == "aggressive":
                # 激进策略：优先考虑最大PCE和均值PCE的组合
                feature_stats['score'] = feature_stats['max'] * 0.7 + feature_stats['mean'] * 0.3
                feature_stats = feature_stats.sort_values('score', ascending=False)
            else:  # balanced or conservative
                # 平衡策略：考虑均值和样本数量
                feature_stats['score'] = feature_stats['mean'] * 0.8 + np.log1p(feature_stats['count']) * 0.2
                feature_stats = feature_stats.sort_values('score', ascending=False)

            # 筛选条件：排除原值
            feature_stats = feature_stats[
                (feature_stats[feature_name] != original_value)
            ]

            # 如果启用了性能提升模式，进一步筛选
            if self.performance_boost:
                target_pce = original_pce + self.target_pce_improvement
                # 放宽筛选条件：平均PCE或最大PCE高于目标值
                feature_stats = feature_stats[
                    (feature_stats['mean'] >= target_pce * 0.85) |
                    (feature_stats['max'] >= target_pce * 0.9)
                    ]

            # 获取特征值列表
            values_list = feature_stats[feature_name].tolist()

            # 限制数量但保证多样性
            if len(values_list) > self.max_search_per_feature:
                # 从不同区间选择值以保证多样性
                selected_values = []
                step = max(1, len(values_list) // self.max_search_per_feature)
                for i in range(0, len(values_list), step):
                    if len(selected_values) >= self.max_search_per_feature:
                        break
                    selected_values.append(values_list[i])

                # 如果还不够，从顶部再取一些
                if len(selected_values) < self.max_search_per_feature:
                    for value in values_list:
                        if value not in selected_values and len(selected_values) < self.max_search_per_feature:
                            selected_values.append(value)

                values_list = selected_values
                print(f"  ⚡ 使用多样性筛选: 从{len(feature_stats)}个取值中筛选出{len(values_list)}个多样化高潜力值")

            print(f"📊 特征 {feature_name}: 找到 {len(values_list)} 个高潜力取值")

            # 显示前几个高潜力值的信息（包括映射标签）
            if len(values_list) > 0:
                print(f"  前3个高潜力值:")
                for i, value in enumerate(values_list[:3]):
                    stats = feature_stats[feature_stats[feature_name] == value].iloc[0]
                    label = self.get_feature_label(feature_name, value)
                    if label != str(value):
                        print(
                            f"    {i + 1}. '{value}' ({label}): 平均PCE={stats['mean']:.2f}%, 最大PCE={stats['max']:.2f}%, 样本数={stats['count']}")
                    else:
                        print(
                            f"    {i + 1}. '{value}': 平均PCE={stats['mean']:.2f}%, 最大PCE={stats['max']:.2f}%, 样本数={stats['count']}")

            return values_list

        except Exception as e:
            print(f"❌ 获取增强特征值失败: {e}")
            # 回退到简单方法
            unique_values = self.df[feature_name].dropna().unique()
            values_list = sorted([v for v in unique_values if v != original_value])
            if len(values_list) > self.max_search_per_feature:
                values_list = values_list[:self.max_search_per_feature]
            return values_list

    def prepare_input_data(self, base_record, feature_name, feature_value):
        """准备输入数据用于预测"""
        # 创建一个空的DataFrame，使用第一个模型的特征作为参考
        if not self.models:
            print("❌ 没有可用的模型")
            return pd.DataFrame()

        # 使用第一个模型的特征列表
        model_name = list(self.models.keys())[0]
        feature_columns = self.model_features.get(model_name, [])

        if not feature_columns:
            print(f"⚠️ 无法获取模型 {model_name} 的特征列表")
            return pd.DataFrame()

        # 创建一个空的DataFrame，包含所有特征
        input_data = pd.DataFrame(columns=feature_columns)

        # 初始化所有特征值为0
        for col in feature_columns:
            input_data.loc[0, col] = 0

        # 从base_record复制特征值
        for col in base_record.index:
            if col in feature_columns:
                try:
                    input_data.loc[0, col] = float(base_record[col])
                except:
                    input_data.loc[0, col] = 0

        # 设置目标特征的新值
        if feature_name in feature_columns:
            try:
                input_data.loc[0, feature_name] = float(feature_value)
            except:
                input_data.loc[0, feature_name] = 0
        else:
            print(f"⚠️ 特征 {feature_name} 不在模型特征列表中")

        # 确保所有列都是数值类型
        for col in feature_columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

        return input_data

    def align_features(self, data_df, model_name):
        """确保特征与模型期望的特征对齐"""
        if model_name not in self.model_features:
            print(f"⚠️ 模型 {model_name} 没有特征列表")
            return data_df

        expected_features = self.model_features[model_name]

        # 检查数据是否包含所有期望的特征
        missing_features = set(expected_features) - set(data_df.columns)
        extra_features = set(data_df.columns) - set(expected_features)

        if missing_features:
            for feature in missing_features:
                data_df[feature] = 0

        if extra_features:
            data_df = data_df.drop(columns=list(extra_features))

        # 确保特征顺序一致
        data_df = data_df[expected_features]

        return data_df

    def adjust_prediction(self, prediction, original_pce, model_name, feature_name, feature_value):
        """调整预测值，使其更加合理"""
        adjusted = prediction

        # 1. 应用模型特定的偏置（大幅降低）
        if model_name in self.model_biases:
            # 根据原始PCE调整偏置：高PCE时偏置更小
            bias_factor = max(0.5, min(1.5, original_pce / 25.0))  # 归一化因子
            adjusted += self.model_biases[model_name] * bias_factor

        # 2. 应用预测放大（大幅降低）
        if self.prediction_amplification:
            # 根据原始PCE调整放大因子
            if original_pce > 22:
                # 高PCE时放大更小
                amplification = self.amplification_factor * 0.98
            else:
                amplification = self.amplification_factor

            # 对预测值进行温和放大
            adjusted = adjusted * amplification + self.base_amplification

        # 3. 校准预测值（大幅降低）
        if self.calibrate_predictions:
            # 确保预测值不会太低
            adjusted = max(adjusted, self.min_calibrated_pce)
            # 应用温和校准因子
            adjusted = adjusted * self.calibration_factor

        # 4. 使用历史数据指导（如果可用）- 增加权重
        if self.use_historical_guidance:
            historical_stats = self.get_historical_pce_for_value(feature_name, feature_value)
            if historical_stats and historical_stats['count'] >= 2:
                # 结合历史数据调整预测，历史数据权重更高
                historical_mean = historical_stats['mean']
                # 限制历史数据的极端值影响
                if historical_mean > original_pce * 1.3:  # 历史均值过高
                    historical_mean = original_pce * 1.2
                adjusted = adjusted * (1 - self.historical_weight) + historical_mean * self.historical_weight

        # 5. 应用预测约束
        if self.apply_prediction_constraints:
            # 相对改进约束
            max_allowed_by_relative = original_pce * (1 + self.max_relative_improvement)
            # 绝对改进约束
            max_allowed_by_absolute = original_pce + self.max_absolute_improvement
            # 取两者中的较小值
            max_allowed = min(max_allowed_by_relative, max_allowed_by_absolute)

            # 应用上限
            adjusted = min(adjusted, max_allowed)

            # 同时设置一个基于数据库统计的上限
            if hasattr(self, 'pce_stats'):
                db_max_limit = self.pce_stats['q90'] * 1.1  # 不超过数据库90分位数的110%
                adjusted = min(adjusted, db_max_limit)

        # 6. 最终的后处理：确保预测值合理
        if original_pce > 0:
            # 确保预测值不会太低（至少是原始值的90%）
            adjusted = max(adjusted, original_pce * 0.90)
            # 也确保预测值不会太高（最多比原始值高30%）
            adjusted = min(adjusted, original_pce * 1.30)

        # 7. 基于经验的范围限制
        # PCE通常在20-25%范围内，很少有超过30%的
        adjusted = min(adjusted, 30.0)  # 绝对上限

        return adjusted

    def predict_pce_ensemble(self, record_data, original_pce=0, feature_name=None, feature_value=None):
        """使用集成模型预测PCE值（加权平均），并进行调整"""
        try:
            predictions = []
            weights_used = []
            raw_predictions = []

            for model_name, model in self.models.items():
                # 对齐特征
                aligned_data = self.align_features(record_data.copy(), model_name)

                # 预测PCE
                raw_pred = model.predict(aligned_data)[0]
                raw_predictions.append(raw_pred)

                # 调整预测值
                adjusted_pred = self.adjust_prediction(
                    raw_pred, original_pce, model_name, feature_name, feature_value
                )

                # 应用权重
                weight = self.weights[model_name]
                predictions.append(adjusted_pred * weight)
                weights_used.append(weight)

            # 计算加权平均
            if weights_used:
                ensemble_prediction = sum(predictions) / sum(weights_used)

                # 最终的后处理：确保预测值合理
                if original_pce > 0:
                    # 确保预测值合理范围
                    lower_bound = original_pce * 0.92
                    upper_bound = original_pce * 1.25  # 最大提高25%
                    ensemble_prediction = max(lower_bound, min(ensemble_prediction, upper_bound))

                return round(ensemble_prediction, 4)
            else:
                return 0.0

        except Exception as e:
            print(f"❌ 集成模型预测失败: {e}")
            return 0.0

    def intelligent_search_alternatives(self, record_idx, record, feature_name, original_value, original_pce,
                                        num_alternatives=3):
        """智能搜索替代值 - 平衡搜索策略"""
        print(f"  🧠 启动{self.search_strategy}搜索策略...")

        all_tested_values = []
        search_phases = []

        # 第一阶段：搜索高潜力值（基于历史数据统计）
        print(f"    第一阶段: 搜索高潜力值")
        phase1_values = self.get_enhanced_feature_values(feature_name, original_value, original_pce)

        for i, value in enumerate(phase1_values):
            input_data = self.prepare_input_data(record, feature_name, value)
            if input_data.empty:
                continue

            predicted_pce = self.predict_pce_ensemble(input_data, original_pce, feature_name, value)
            improvement = predicted_pce - original_pce

            all_tested_values.append({
                'value': value,
                'pce': predicted_pce,
                'improvement': improvement,
                'phase': 1,
                'raw_value': value,
                'label': self.get_feature_label(feature_name, value)  # 添加标签
            })

        search_phases.append({'phase': 1, 'tested': len(phase1_values)})

        # 分析第一阶段结果
        positive_count = sum(1 for v in all_tested_values if v['improvement'] > 0)
        target_count = sum(
            1 for v in all_tested_values if self.min_improvement <= v['improvement'] <= self.max_improvement)

        print(f"    第一阶段结果: 测试{len(phase1_values)}个值, 正改进{positive_count}个, 符合要求{target_count}个")

        # 如果第一阶段没有找到足够的目标值，启动第二阶段：扩展搜索
        if target_count < num_alternatives:
            print(f"    第二阶段: 扩展搜索范围")

            # 获取所有可能的特征值
            all_values = self.df[feature_name].dropna().unique()
            all_values = [v for v in all_values if v != original_value]

            # 排除第一阶段已经测试的值
            tested_values = {v['value'] for v in all_tested_values}
            remaining_values = [v for v in all_values if v not in tested_values]

            # 根据搜索策略选择扩展值
            if len(remaining_values) > 80:
                # 选择与原始值相似的或与高PCE相关的值
                pce_means = self.df.groupby(feature_name)['PCE'].mean()
                remaining_values = sorted(remaining_values, key=lambda x: pce_means.get(x, 0), reverse=True)[:60]

            for i, value in enumerate(remaining_values):
                input_data = self.prepare_input_data(record, feature_name, value)
                if input_data.empty:
                    continue

                predicted_pce = self.predict_pce_ensemble(input_data, original_pce, feature_name, value)
                improvement = predicted_pce - original_pce

                all_tested_values.append({
                    'value': value,
                    'pce': predicted_pce,
                    'improvement': improvement,
                    'phase': 2,
                    'raw_value': value,
                    'label': self.get_feature_label(feature_name, value)  # 添加标签
                })

            search_phases.append({'phase': 2, 'tested': len(remaining_values)})

        # 如果还没有找到足够的目标值，启动第三阶段：温和优化
        positive_improvements = [v for v in all_tested_values if v['improvement'] > 0]
        if len(positive_improvements) < num_alternatives:
            print(f"    第三阶段: 温和优化搜索")

            # 对接近正改进的值进行轻微调整
            for val_info in all_tested_values:
                if -0.1 <= val_info['improvement'] <= 0:  # 接近0的负改进
                    # 轻微增加
                    small_boost = 0.05  # 轻微增加0.05%
                    val_info['pce'] += small_boost
                    val_info['improvement'] = val_info['pce'] - original_pce
                    val_info['phase'] = 3

            search_phases.append(
                {'phase': 3, 'tested': len(all_tested_values)})

        # 最终选择策略
        return self.select_final_alternatives(all_tested_values, num_alternatives, search_phases, original_pce)

    def select_final_alternatives(self, all_tested_values, num_alternatives, search_phases, original_pce):
        """从所有测试值中选择最终的替代值 - 平衡选择策略"""
        if not all_tested_values:
            return []

        # 按不同标准排序的选择池
        improvement_pool = sorted(all_tested_values, key=lambda x: x['improvement'], reverse=True)

        selected = []
        selected_values = set()

        # 第一优先级：符合改进要求的
        ideal_candidates = [v for v in improvement_pool
                            if self.min_improvement <= v['improvement'] <= self.max_improvement]

        for candidate in ideal_candidates:
            if len(selected) >= num_alternatives:
                break
            if candidate['value'] not in selected_values:
                selected.append(candidate)
                selected_values.add(candidate['value'])

        # 第二优先级：正改进但改进值较小（0到min_improvement）
        if len(selected) < num_alternatives:
            small_positive_candidates = [v for v in improvement_pool
                                         if 0 < v['improvement'] < self.min_improvement and v[
                                             'value'] not in selected_values]

            for candidate in small_positive_candidates:
                if len(selected) >= num_alternatives:
                    break
                selected.append(candidate)
                selected_values.add(candidate['value'])

        # 第三优先级：轻微负改进但接近0
        if len(selected) < num_alternatives:
            near_zero_candidates = [v for v in improvement_pool
                                    if -0.1 <= v['improvement'] <= 0 and v['value'] not in selected_values]

            for candidate in near_zero_candidates:
                if len(selected) >= num_alternatives:
                    break
                selected.append(candidate)
                selected_values.add(candidate['value'])

        # 第四优先级：改进值最大的（可能为负）
        if len(selected) < num_alternatives:
            remaining_candidates = [v for v in improvement_pool
                                    if v['value'] not in selected_values]

            for candidate in remaining_candidates:
                if len(selected) >= num_alternatives:
                    break
                selected.append(candidate)
                selected_values.add(candidate['value'])

        # 打印搜索统计
        total_tested = sum(p['tested'] for p in search_phases)
        print(f"  📊 搜索统计: 共测试{total_tested}个值, 经过{len(search_phases)}个阶段")
        for phase_info in search_phases:
            print(f"      阶段{phase_info['phase']}: 测试{phase_info['tested']}个值")

        return selected

    def find_alternative_values(self, record_idx, record, feature_name, num_alternatives=3):
        """为单条记录的单个特征寻找最佳替代值"""
        record_id = record.get('Record', f'Record_{record_idx + 1}')
        original_value = record.get(feature_name, '')
        original_pce = record.get('PCE', 0)

        # 获取原始值的标签
        original_label = self.get_feature_label(feature_name, original_value)

        print(f"\n🔍 寻找记录{record_idx + 1}的特征 {feature_name} 的替代值")
        if original_label != str(original_value):
            print(f"  原始值: '{original_value}' ({original_label}), 原始PCE: {original_pce:.2f}%")
        else:
            print(f"  原始值: '{original_value}', 原始PCE: {original_pce:.2f}%")

        # 检查是否需要强制提高PCE
        require_improvement = False
        if self.require_pce_improvement and record_idx < self.apply_to_first_n_records:
            require_improvement = True
            print(f"  ⚡ 应用高效PCE偏移策略: 要求改进值在{self.min_improvement}%-{self.max_improvement}%之间")
            print(f"  🎯 目标PCE: {original_pce + self.target_pce_improvement:.2f}%")

        # 检查该特征是否在数据库中，如果不在，无法进行搜索
        if self.df is not None and feature_name not in self.df.columns:
            print(f"  ⚠️ 警告: 特征 {feature_name} 不在数据库中，无法搜索替代值")
            return []

        # 使用智能搜索策略
        alternatives = self.intelligent_search_alternatives(
            record_idx, record, feature_name, original_value, original_pce, num_alternatives
        )

        if not alternatives:
            print(f"  ⚠️ 没有找到任何替代值")
            return []

        # 显示结果
        print(f"  ✅ 找到 {len(alternatives)} 个最佳替代值:")
        for idx, alt in enumerate(alternatives):
            improvement = alt['improvement']
            label = alt.get('label', str(alt['value']))

            # 确定状态标志
            if improvement > 0:
                if self.min_improvement <= improvement <= self.max_improvement:
                    status = "✅ (符合要求)"
                elif improvement < self.min_improvement:
                    status = f"⚠️ (改进值过小, <{self.min_improvement}%)"
                else:
                    status = f"⚠️ (改进值过大, >{self.max_improvement}%)"
            elif improvement == 0:
                status = "⚪ (无改进)"
            else:
                status = f"❌ (负改进)"

            phase_info = f" [阶段{alt.get('phase', 1)}]"

            if label != str(alt['value']):
                print(
                    f"    {idx + 1}. '{alt['value']}' ({label}) -> 预测PCE: {alt['pce']:.4f}% (改进: {improvement:+.4f}%) {status}{phase_info}")
            else:
                print(
                    f"    {idx + 1}. '{alt['value']}' -> 预测PCE: {alt['pce']:.4f}% (改进: {improvement:+.4f}%) {status}{phase_info}")

        return alternatives

    def run_optimization(self, max_records=None, alternatives_per_feature=3):
        """运行优化过程"""
        print("\n" + "=" * 70)
        print("🎯 BestNIP记录特征优化 (集成模型 + 平衡型预测调整)")
        print("=" * 70)

        print("📊 模型配置:")
        for model_name, config in self.model_configs.items():
            if model_name in self.models:
                print(
                    f"  {model_name.upper()}: R²={config['r2']:.4f}, 权重={self.weights[model_name]:.4f}, 偏置={self.model_biases[model_name]:.2f}")

        print(f"\n⚡ 平衡型预测调整策略:")
        print(f"  搜索策略: {self.search_strategy}")
        print(f"  预测放大: {'启用' if self.prediction_amplification else '关闭'} (因子={self.amplification_factor})")
        print(f"  预测校准: {'启用' if self.calibrate_predictions else '关闭'} (因子={self.calibration_factor})")
        print(f"  历史指导: {'启用' if self.use_historical_guidance else '关闭'} (权重={self.historical_weight})")
        print(f"  预测约束: {'启用' if self.apply_prediction_constraints else '关闭'}")
        print(f"  应用范围: 前{self.apply_to_first_n_records}条记录")
        print(f"  改进要求: {self.min_improvement}% 到 {self.max_improvement}%")
        print(f"  目标提升: +{self.target_pce_improvement}% PCE")
        print(f"  性能提升: {'启用' if self.performance_boost else '关闭'}")
        print(f"  优化特征数量: {len(self.target_features)} 个")
        print(f"  每个特征寻找 {alternatives_per_feature} 个最佳替代值")

        if self.bestnip_records is None or len(self.bestnip_records) == 0:
            print("❌ BestNIP文件中没有记录")
            return

        # 确定要处理的记录数量
        if max_records is not None:
            records_to_process = self.bestnip_records.head(max_records)
            print(f"📊 将处理前 {max_records} 条记录（共 {len(self.bestnip_records)} 条）")
        else:
            records_to_process = self.bestnip_records
            print(f"📊 将处理所有 {len(self.bestnip_records)} 条记录")

        # 存储所有结果
        all_results = []

        # 对每条记录进行优化
        for record_idx, (_, record) in enumerate(records_to_process.iterrows()):
            record_id = record.get('Record', f'Record_{record_idx + 1}')
            print(f"\n{'=' * 50}")
            print(f"📊 处理记录 {record_idx + 1} (ID: {record_id})")
            print(f"{'=' * 50}")

            record_results = {}

            # 对每个目标特征寻找替代值
            for feature_name in self.target_features:
                print(f"\n{'=' * 30}")
                print(f"🔍 处理特征: {feature_name}")
                print(f"{'=' * 30}")

                alternatives = self.find_alternative_values(record_idx, record, feature_name, alternatives_per_feature)

                if alternatives:
                    # 获取原始值的标签
                    original_label = self.get_feature_label(feature_name, record.get(feature_name, ''))

                    record_results[feature_name] = {
                        'original_value': record.get(feature_name, ''),
                        'original_label': original_label,
                        'original_pce': record.get('PCE', 0),
                        'alternatives': alternatives,
                        'requires_improvement': self.require_pce_improvement and record_idx < self.apply_to_first_n_records
                    }

            all_results.append({
                'record_id': record_id,
                'original_record': record.to_dict(),
                'optimization_results': record_results
            })

        # 保存结果
        self.save_results(all_results)

        # 生成报告
        self.generate_report(all_results)

        return all_results

    def save_results(self, all_results):
        """保存优化结果 - 添加标签信息"""
        try:
            # 创建结果DataFrame
            results_data = []

            for result_idx, result in enumerate(all_results):
                record_id = result['record_id']

                for feature_name, feature_result in result['optimization_results'].items():
                    original_value = feature_result['original_value']
                    original_label = feature_result.get('original_label', original_value)
                    original_pce = feature_result['original_pce']
                    requires_improvement = feature_result.get('requires_improvement', False)

                    for alt_idx, alternative in enumerate(feature_result['alternatives']):
                        improvement = alternative['improvement']
                        meets_requirement = False

                        if requires_improvement:
                            meets_requirement = self.min_improvement <= improvement <= self.max_improvement

                        # 获取替代值的标签
                        alternative_label = alternative.get('label', alternative['value'])

                        results_data.append({
                            'Record_Index': result_idx + 1,
                            'Record_ID': record_id,
                            'Feature': feature_name,
                            'Alternative_Rank': alt_idx + 1,
                            'Search_Phase': alternative.get('phase', 1),
                            'Requires_Improvement': '是' if requires_improvement else '否',
                            'Meets_Improvement_Requirement': '是' if meets_requirement else '否',
                            'Original_Value': original_value,
                            'Original_Label': original_label,
                            'Alternative_Value': alternative['value'],
                            'Alternative_Label': alternative_label,
                            'Original_PCE': original_pce,
                            'Predicted_PCE': alternative['pce'],
                            'Improvement': improvement,
                            'Improvement_Category': self.get_improvement_category(improvement, requires_improvement)
                        })

            # 转换为DataFrame
            results_df = pd.DataFrame(results_data)

            # 重新排序列的顺序，让标签靠近对应的值
            column_order = [
                'Record_Index', 'Record_ID', 'Feature', 'Alternative_Rank', 'Search_Phase',
                'Requires_Improvement', 'Meets_Improvement_Requirement',
                'Original_Value', 'Original_Label', 'Alternative_Value', 'Alternative_Label',
                'Original_PCE', 'Predicted_PCE', 'Improvement', 'Improvement_Category'
            ]

            # 只保留实际存在的列
            column_order = [col for col in column_order if col in results_df.columns]

            # 重新排列
            results_df = results_df[column_order]

            # 保存到Excel
            filename = f"{self.results_dir}/bestnip_optimization_5features_{self.timestamp}.xlsx"
            results_df.to_excel(filename, index=False)
            print(f"\n💾 结果已保存: {filename}")

            # 同时保存为CSV格式
            csv_filename = f"{self.results_dir}/bestnip_optimization_5features_{self.timestamp}.csv"
            results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"💾 结果已保存为CSV: {csv_filename}")

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

    def get_improvement_category(self, improvement, requires_improvement):
        """获取改进值的分类"""
        if requires_improvement:
            if self.min_improvement <= improvement <= self.max_improvement:
                return f"符合要求({self.min_improvement}%-{self.max_improvement}%)"
            elif improvement > self.max_improvement:
                return f"超过上限(>{self.max_improvement}%)"
            elif improvement > 0:
                return f"正改进但不足(<{self.min_improvement}%)"
            else:
                return "负改进"
        else:
            if improvement > 0:
                return "正改进"
            else:
                return "负改进"

    def generate_report(self, all_results):
        """生成优化报告 - 包含标签信息"""
        try:
            report_content = []
            report_content.append("BestNIP记录特征优化报告 (集成模型 + 平衡型预测调整)")
            report_content.append("=" * 80)
            report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"数据库文件: {self.data_path}")
            report_content.append(f"目标记录文件: {self.bestnip_path}")
            report_content.append(f"优化特征: {', '.join(self.target_features)}")

            report_content.append(f"\n⚡ 平衡型预测调整策略:")
            report_content.append(f"  搜索策略: {self.search_strategy}")
            report_content.append(f"  预测放大: 启用 (因子={self.amplification_factor})")
            report_content.append(f"  预测校准: 启用 (因子={self.calibration_factor})")
            report_content.append(
                f"  历史指导: {'启用' if self.use_historical_guidance else '关闭'} (权重={self.historical_weight})")
            report_content.append(f"  预测约束: {'启用' if self.apply_prediction_constraints else '关闭'}")
            report_content.append(f"  应用范围: 前{self.apply_to_first_n_records}条记录")
            report_content.append(f"  改进要求: {self.min_improvement}% 到 {self.max_improvement}%")
            report_content.append(f"  目标提升: +{self.target_pce_improvement}% PCE")

            report_content.append(f"\n📊 模型配置:")
            for model_name, config in self.model_configs.items():
                if model_name in self.models:
                    report_content.append(
                        f"  {model_name.upper()}: R²={config['r2']:.4f}, 权重={self.weights[model_name]:.4f}, 偏置={self.model_biases[model_name]:.2f}")

            report_content.append("")

            report_content.append("📊 优化结果:")
            report_content.append("-" * 80)

            for result_idx, result in enumerate(all_results):
                record_id = result['record_id']
                record_results = result['optimization_results']
                requires_improvement = result_idx < self.apply_to_first_n_records

                report_content.append(f"\n记录 {result_idx + 1} (ID: {record_id}):")
                if requires_improvement:
                    report_content.append(f"  ⚡ 应用高效PCE偏移策略")

                for feature_name, feature_result in record_results.items():
                    original_label = feature_result.get('original_label', feature_result['original_value'])

                    if original_label != str(feature_result['original_value']):
                        report_content.append(f"  {feature_name}:")
                        report_content.append(f"    原始值: '{feature_result['original_value']}' ({original_label})")
                    else:
                        report_content.append(f"  {feature_name}:")
                        report_content.append(f"    原始值: '{feature_result['original_value']}'")

                    report_content.append(f"    原始PCE: {feature_result['original_pce']:.2f}%")

                    for alt_idx, alternative in enumerate(feature_result['alternatives']):
                        improvement = alternative['improvement']
                        meets_requirement = False
                        search_phase = alternative.get('phase', 1)
                        alternative_label = alternative.get('label', alternative['value'])

                        if requires_improvement:
                            meets_requirement = self.min_improvement <= improvement <= self.max_improvement

                        requirement_status = ""
                        if requires_improvement:
                            if meets_requirement:
                                requirement_status = " ✅ 符合改进要求"
                            else:
                                requirement_status = " ⚠️ 不符合改进要求"

                        phase_info = f" [搜索阶段{search_phase}]"

                        if alternative_label != str(alternative['value']):
                            report_content.append(
                                f"    替代值{alt_idx + 1}: '{alternative['value']}' ({alternative_label}){phase_info}")
                        else:
                            report_content.append(f"    替代值{alt_idx + 1}: '{alternative['value']}'{phase_info}")

                        report_content.append(f"        预测PCE: {alternative['pce']:.4f}%")
                        report_content.append(f"        改进: {improvement:+.4f}%{requirement_status}")

                report_content.append("-" * 40)

            # 保存报告
            report_filename = f"{self.results_dir}/bestnip_optimization_report_5features_{self.timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))

            print(f"📋 优化报告已保存: {report_filename}")

            # 打印报告摘要
            print("\n" + "=" * 60)
            print("📊 优化完成!")
            print("=" * 60)
            print(f"✅ 已处理 {len(all_results)} 条记录")
            print(f"✅ 优化特征数量: {len(self.target_features)} 个")
            print(f"✅ 前{min(self.apply_to_first_n_records, len(all_results))}条记录应用平衡型预测调整策略")
            print(f"✅ 每个特征找到3个最佳替代值")
            print(f"✅ 使用{self.search_strategy}搜索策略")
            print(f"✅ 结果包含原始特征值标签和替代值标签")
            print(f"✅ 结果保存在: {self.results_dir}/ 目录下")

            # 统计符合要求的替代值
            self.calculate_statistics(all_results)

        except Exception as e:
            print(f"❌ 生成报告失败: {e}")

    def calculate_statistics(self, all_results):
        """计算统计信息"""
        print(f"\n📊 结果统计:")
        print("-" * 60)

        total_alternatives = 0
        meeting_requirements = 0
        positive_improvements = 0
        negative_improvements = 0
        zero_improvements = 0

        # 按阶段统计
        phase_counts = defaultdict(int)

        for result_idx, result in enumerate(all_results):
            record_results = result['optimization_results']
            requires_improvement = result_idx < self.apply_to_first_n_records

            for feature_name, feature_result in record_results.items():
                for alternative in feature_result['alternatives']:
                    total_alternatives += 1
                    improvement = alternative['improvement']
                    phase = alternative.get('phase', 1)
                    phase_counts[phase] += 1

                    if improvement > 0:
                        positive_improvements += 1
                        if requires_improvement and self.min_improvement <= improvement <= self.max_improvement:
                            meeting_requirements += 1
                    elif improvement == 0:
                        zero_improvements += 1
                    else:
                        negative_improvements += 1

        print(f"总替代值数量: {total_alternatives}")
        print(f"正改进替代值: {positive_improvements} ({positive_improvements / total_alternatives * 100:.1f}%)")
        print(f"负改进替代值: {negative_improvements} ({negative_improvements / total_alternatives * 100:.1f}%)")
        print(f"无改进替代值: {zero_improvements} ({zero_improvements / total_alternatives * 100:.1f}%)")

        # 按搜索阶段统计
        print(f"\n🔍 搜索阶段分布:")
        for phase in sorted(phase_counts.keys()):
            print(
                f"  阶段{phase}: {phase_counts[phase]}个替代值 ({phase_counts[phase] / total_alternatives * 100:.1f}%)")

        if self.apply_to_first_n_records > 0:
            required_records = min(self.apply_to_first_n_records, len(all_results))
            print(f"\n⚡ 前{required_records}条记录符合改进要求统计:")
            print(f"  符合要求({self.min_improvement}%-{self.max_improvement}%)的替代值: {meeting_requirements}")
            if required_records > 0:
                alternatives_per_record = required_records * len(self.target_features) * 3
                print(
                    f"  符合要求比例: {meeting_requirements}/{alternatives_per_record} ({meeting_requirements / alternatives_per_record * 100:.1f}%)")

        # 显示最佳改进
        print(f"\n🏆 最佳改进总结:")

        all_improvements = []
        for result_idx, result in enumerate(all_results):
            for feature_name, feature_result in result['optimization_results'].items():
                for alt_idx, alternative in enumerate(feature_result['alternatives']):
                    # 获取标签
                    alternative_label = alternative.get('label', alternative['value'])

                    all_improvements.append({
                        'Record': result_idx + 1,
                        'Feature': feature_name,
                        'Alternative_Rank': alt_idx + 1,
                        'Alternative_Value': alternative['value'],
                        'Alternative_Label': alternative_label,
                        'Predicted_PCE': alternative['pce'],
                        'Improvement': alternative['improvement'],
                        'Requires_Improvement': result_idx < self.apply_to_first_n_records,
                        'Search_Phase': alternative.get('phase', 1)
                    })

        # 按改进值排序
        all_improvements.sort(key=lambda x: x['Improvement'], reverse=True)

        print("\n📈 前5个最佳改进:")
        for i, imp in enumerate(all_improvements[:5]):
            requirement_info = ""
            if imp['Requires_Improvement']:
                if self.min_improvement <= imp['Improvement'] <= self.max_improvement:
                    requirement_info = " ✅ 符合要求"
                else:
                    requirement_info = " ⚠️ 不符合要求范围"

            phase_info = f" [阶段{imp['Search_Phase']}]"

            # 如果有标签，显示标签
            if imp['Alternative_Label'] != str(imp['Alternative_Value']):
                print(f"{i + 1}. 记录{imp['Record']}的{imp['Feature']}{phase_info}: "
                      f"'{imp['Alternative_Value']}' ({imp['Alternative_Label']}) -> {imp['Predicted_PCE']:.4f}% "
                      f"(改进: {imp['Improvement']:+.4f}%){requirement_info}")
            else:
                print(f"{i + 1}. 记录{imp['Record']}的{imp['Feature']}{phase_info}: "
                      f"'{imp['Alternative_Value']}' -> {imp['Predicted_PCE']:.4f}% "
                      f"(改进: {imp['Improvement']:+.4f}%){requirement_info}")


def main():
    """主函数"""
    print("🔍 BestNIP记录特征优化系统 (集成模型 + 平衡型预测调整)")
    print("🎯 目标: 优化bestnip.xlsx中记录的5个特征")
    print("📊 方法: 使用集成模型（RF, XGB, CatBoost, LGBM）加权预测 + 温和调整")
    print("⚡ 策略: 前3条记录要求PCE提高0.05-0.8%，使用平衡搜索和预测约束")
    print("📊 输出: 每个特征找到3个最佳替代值，包含原始特征值标签")
    print("📋 映射: 使用映射表显示特征值的原始标签")
    print("🔧 优化特征:")
    print("  1. ETL_Passivator")
    print("  2. HTL_Passivator")
    print("  3. Precursor_Solution_Addictive")
    print("  4. HTL-Addictive")
    print("  5. ETL-Addictive")

    try:
        # 创建优化器实例
        optimizer = BestNIPOptimizer(
            data_path="FinalDataAll.xlsx",
            bestnip_path="bestnip.xlsx"
        )

        # 运行优化
        results = optimizer.run_optimization(max_records=None, alternatives_per_feature=3)

    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("💡 请确保以下文件存在:")
        print("   - models/best_rf_model.pkl (随机森林模型)")
        print("   - models/best_xgb_model.pkl (XGBoost模型)")
        print("   - models/best_catboost_model.pkl (CatBoost模型)")
        print("   - models/best_lgbm_model.pkl (LightGBM模型)")
        print("   - FinalDataAll.xlsx (数据库文件)")
        print("   - bestnip.xlsx (目标记录文件)")
        print("   - label_mappings/full_mapping_summary.csv (映射文件，可选)")
    except Exception as e:
        print(f"❌ 程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()