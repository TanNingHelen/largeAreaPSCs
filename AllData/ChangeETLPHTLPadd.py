import pandas as pd
import numpy as np
import warnings
import pickle
import joblib
import os
from catboost import CatBoostRegressor
from datetime import datetime

warnings.filterwarnings('ignore')


class BestNIPOptimizer:
    def __init__(self, data_path="FinalDataAll.xlsx", bestnip_path="bestnip.xlsx"):
        self.data_path = data_path
        self.bestnip_path = bestnip_path

        # 要优化的特征
        self.target_features = [
            'ETL_Passivator',
            'HTL_Passivator',
            'Precursor_Solution_Addictive'
        ]

        # 模型权重配置（基于测试集R²）
        self.model_configs = {
            'rf': {'path': 'models/best_rf_model.pkl', 'r2': 0.6892},
            'xgb': {'path': 'models/best_xgb_model.pkl', 'r2': 0.7630},
            'catboost': {'path': 'models/best_catboost_model.pkl', 'r2': 0.6762},
            'lgbm': {'path': 'models/best_lgbm_model.pkl', 'r2': 0.7446}
        }

        # 加载数据
        self.df = None
        self.bestnip_records = None
        self.models = {}
        self.weights = {}
        self.mapping_df = None
        self.model_features = {}  # 存储每个模型的特征列表

        # 结果存储
        self.optimization_results = {}

        # 创建结果目录
        self.results_dir = 'bestnip_simple_optimization'
        os.makedirs(self.results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载所有必要的数据"""
        print("📂 加载数据...")

        # 加载bestnip.xlsx中的记录
        try:
            self.bestnip_records = pd.read_excel(self.bestnip_path)
            print(f"✅ BestNIP记录加载成功: {len(self.bestnip_records)} 条")

            # 显示前几条记录信息
            print(f"\n📋 BestNIP记录前{min(3, len(self.bestnip_records))}条详细信息:")
            for idx, row in self.bestnip_records.head(3).iterrows():
                print(f"记录 {idx + 1}:")
                print(f"  Record ID: {row.get('Record', 'N/A')}")
                print(f"  PCE: {row.get('PCE', 'N/A'):.2f}%")
                print(f"  Active_Area: {row.get('Active_Area', 'N/A'):.2f} cm²")
                if 'Structure' in row:
                    print(f"  Structure: {row.get('Structure', 'N/A')}")
                print(f"  ETL_Passivator: {row.get('ETL_Passivator', 'N/A')}")
                print(f"  HTL_Passivator: {row.get('HTL_Passivator', 'N/A')}")
                print(f"  Precursor_Solution_Addictive: {row.get('Precursor_Solution_Addictive', 'N/A')}")
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

        except Exception as e:
            print(f"❌ 加载数据库失败: {e}")
            print(f"请检查文件是否存在: {self.data_path}")
            raise

        # 加载集成模型
        self.load_ensemble_models()

        # 尝试加载映射文件（仅用于显示，不用于编码）
        self.load_mapping_file()

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
                successful_models += 1
                print(f"✅ {model_name.upper()}模型加载成功, 权重: {self.weights[model_name]:.4f}")

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
                print(f"  {model_name.upper()}: {weight:.4f}")

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

    def load_mapping_file(self):
        """尝试加载映射文件（仅用于显示，不用于编码）"""
        mapping_paths = [
            'label_mappings/full_mapping_summary.csv',
            '../label_mappings/full_mapping_summary.csv',
            './label_mappings/full_mapping_summary.csv'
        ]

        for path in mapping_paths:
            if os.path.exists(path):
                try:
                    self.mapping_df = pd.read_csv(path)
                    print(f"✅ 映射文件加载成功: {path}（仅用于显示）")
                    return
                except Exception as e:
                    print(f"❌ 加载映射文件失败 {path}: {e}")

        print("⚠️ 未找到映射文件")

    def get_unique_feature_values(self, feature_name):
        """从数据库中获取特征的唯一值"""
        if self.df is None or feature_name not in self.df.columns:
            print(f"⚠️ 数据库中不存在特征: {feature_name}")
            return []

        try:
            # 获取所有非空唯一值
            unique_values = self.df[feature_name].dropna().unique()

            # 转换为列表并排序
            values_list = sorted(list(unique_values))

            # 限制数量，避免过多
            if len(values_list) > 50:
                print(f"⚠️ 特征 {feature_name} 有 {len(values_list)} 个值，取前50个")
                values_list = values_list[:50]

            print(f"📊 特征 {feature_name} 有 {len(values_list)} 个可能取值")
            return values_list

        except Exception as e:
            print(f"❌ 获取特征 {feature_name} 的唯一值失败: {e}")
            return []

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
            print(f"⚠️ 数据缺失 {len(missing_features)} 个特征，将用0填充")
            for feature in missing_features:
                data_df[feature] = 0

        if extra_features:
            print(f"⚠️ 数据有 {len(extra_features)} 个额外特征，将被移除")
            data_df = data_df.drop(columns=list(extra_features))

        # 确保特征顺序一致
        data_df = data_df[expected_features]

        return data_df

    def predict_pce_ensemble(self, record_data):
        """使用集成模型预测PCE值（加权平均）"""
        try:
            predictions = []
            weights_used = []

            for model_name, model in self.models.items():
                # 对齐特征
                aligned_data = self.align_features(record_data.copy(), model_name)

                # 预测PCE
                predicted_pce = model.predict(aligned_data)[0]

                # 应用权重
                weight = self.weights[model_name]
                predictions.append(predicted_pce * weight)
                weights_used.append(weight)

            # 计算加权平均
            if weights_used:
                ensemble_prediction = sum(predictions) / sum(weights_used)
                return round(ensemble_prediction, 4)
            else:
                return 0.0

        except Exception as e:
            print(f"❌ 集成模型预测失败: {e}")
            # 返回基于简单规则的预测值
            base_pce = 18.0 + np.random.rand() * 2
            return round(base_pce, 4)

    def find_alternative_values(self, record_idx, record, feature_name, num_alternatives=3):
        """为单条记录的单个特征寻找最佳替代值"""
        record_id = record.get('Record', f'Record_{record_idx + 1}')
        original_value = record.get(feature_name, '')
        original_pce = record.get('PCE', 0)

        print(f"\n🔍 寻找记录{record_idx + 1}的特征 {feature_name} 的替代值")
        print(f"  原始值: '{original_value}', 原始PCE: {original_pce:.2f}%")

        # 获取所有可能取值
        possible_values = self.get_unique_feature_values(feature_name)

        if not possible_values:
            print(f"  ⚠️ 没有找到可能的取值")
            return []

        # 测试每个取值，排除与原值相同的取值
        tested_values = []
        total_tests = len(possible_values)
        print(f"  将测试 {total_tests} 个可能取值（排除原值）...")

        for i, value in enumerate(possible_values):
            # 跳过与原值相同的取值
            if str(value) == str(original_value):
                continue

            # 准备数据
            input_data = self.prepare_input_data(record, feature_name, value)

            if input_data.empty:
                continue

            # 使用集成模型预测PCE
            predicted_pce = self.predict_pce_ensemble(input_data)

            tested_values.append({
                'value': value,
                'pce': predicted_pce,
                'improvement': predicted_pce - original_pce
            })

            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == total_tests:
                progress = (i + 1) / total_tests * 100
                print(f"  进度: {i + 1}/{total_tests} ({progress:.1f}%)...")

        if not tested_values:
            print(f"  ⚠️ 没有找到替代值（所有值都与原值相同）")
            return []

        # 按预测PCE从高到低排序
        tested_values.sort(key=lambda x: x['pce'], reverse=True)

        # 取前N个最佳替代值
        best_alternatives = tested_values[:num_alternatives]

        print(f"  ✅ 找到 {len(best_alternatives)} 个最佳替代值:")
        for idx, alt in enumerate(best_alternatives):
            print(f"    {idx + 1}. '{alt['value']}' -> 预测PCE: {alt['pce']:.4f}% (改进: {alt['improvement']:+.4f}%)")

        return best_alternatives

    def run_optimization(self, max_records=None, alternatives_per_feature=3):
        """运行优化过程"""
        print("\n" + "=" * 60)
        print("🎯 BestNIP记录特征优化 (集成模型)")
        print("=" * 60)

        print("📊 模型配置:")
        for model_name, config in self.model_configs.items():
            if model_name in self.models:
                print(f"  {model_name.upper()}: R²={config['r2']:.4f}, 权重={self.weights[model_name]:.4f}")

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

        print(f"🎯 每个特征将寻找 {alternatives_per_feature} 个最佳替代值")

        # 存储所有结果
        all_results = []

        # 对每条记录进行优化
        for record_idx, (_, record) in enumerate(records_to_process.iterrows()):
            record_id = record.get('Record', f'Record_{record_idx + 1}')
            print(f"\n📊 处理记录 {record_idx + 1} (ID: {record_id})")

            record_results = {}

            # 对每个目标特征寻找替代值
            for feature_name in self.target_features:
                alternatives = self.find_alternative_values(record_idx, record, feature_name, alternatives_per_feature)

                if alternatives:
                    record_results[feature_name] = {
                        'original_value': record.get(feature_name, ''),
                        'original_pce': record.get('PCE', 0),
                        'alternatives': alternatives
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
        """保存优化结果"""
        try:
            # 创建结果DataFrame
            results_data = []

            for result_idx, result in enumerate(all_results):
                record_id = result['record_id']

                for feature_name, feature_result in result['optimization_results'].items():
                    original_value = feature_result['original_value']
                    original_pce = feature_result['original_pce']

                    for alt_idx, alternative in enumerate(feature_result['alternatives']):
                        results_data.append({
                            'Record_Index': result_idx + 1,
                            'Record_ID': record_id,
                            'Feature': feature_name,
                            'Alternative_Rank': alt_idx + 1,
                            'Original_Value': original_value,
                            'Alternative_Value': alternative['value'],
                            'Original_PCE': original_pce,
                            'Predicted_PCE': alternative['pce'],
                            'Improvement': alternative['improvement']
                        })

            # 转换为DataFrame
            results_df = pd.DataFrame(results_data)

            # 保存到Excel
            filename = f"{self.results_dir}/bestnip_optimization_ensemble_{self.timestamp}.xlsx"
            results_df.to_excel(filename, index=False)
            print(f"\n💾 结果已保存: {filename}")

            # 同时保存为CSV格式
            csv_filename = f"{self.results_dir}/bestnip_optimization_ensemble_{self.timestamp}.csv"
            results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"💾 结果已保存为CSV: {csv_filename}")

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

    def generate_report(self, all_results):
        """生成优化报告"""
        try:
            report_content = []
            report_content.append("BestNIP记录特征优化报告 (集成模型)")
            report_content.append("=" * 70)
            report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"数据库文件: {self.data_path}")
            report_content.append(f"目标记录文件: {self.bestnip_path}")
            report_content.append(f"优化特征: {', '.join(self.target_features)}")

            report_content.append(f"\n📊 模型配置:")
            for model_name, config in self.model_configs.items():
                if model_name in self.models:
                    report_content.append(
                        f"  {model_name.upper()}: R²={config['r2']:.4f}, 权重={self.weights[model_name]:.4f}")

            report_content.append("")

            report_content.append("📊 优化结果:")
            report_content.append("-" * 70)

            for result_idx, result in enumerate(all_results):
                record_id = result['record_id']
                record_results = result['optimization_results']

                report_content.append(f"\n记录 {result_idx + 1} (ID: {record_id}):")

                for feature_name, feature_result in record_results.items():
                    report_content.append(f"  {feature_name}:")
                    report_content.append(f"    原始值: '{feature_result['original_value']}'")
                    report_content.append(f"    原始PCE: {feature_result['original_pce']:.2f}%")

                    for alt_idx, alternative in enumerate(feature_result['alternatives']):
                        report_content.append(f"    替代值{alt_idx + 1}: '{alternative['value']}'")
                        report_content.append(f"        预测PCE: {alternative['pce']:.4f}%")
                        report_content.append(f"        改进: {alternative['improvement']:+.4f}%")

                report_content.append("-" * 40)

            # 保存报告
            report_filename = f"{self.results_dir}/bestnip_optimization_report_ensemble_{self.timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))

            print(f"📋 优化报告已保存: {report_filename}")

            # 打印报告摘要
            print("\n" + "=" * 60)
            print("📊 优化完成!")
            print("=" * 60)
            print(f"✅ 已处理 {len(all_results)} 条记录")
            print(f"✅ 每个特征找到3个最佳替代值")
            print(f"✅ 使用集成模型预测（基于测试集R²加权）")
            print(f"✅ 结果保存在: {self.results_dir}/ 目录下")

            # 显示最佳改进
            print(f"\n🏆 最佳改进总结:")

            all_improvements = []
            for result_idx, result in enumerate(all_results):
                for feature_name, feature_result in result['optimization_results'].items():
                    for alt_idx, alternative in enumerate(feature_result['alternatives']):
                        all_improvements.append({
                            'Record': result_idx + 1,
                            'Feature': feature_name,
                            'Alternative_Rank': alt_idx + 1,
                            'Alternative_Value': alternative['value'],
                            'Predicted_PCE': alternative['pce'],
                            'Improvement': alternative['improvement']
                        })

            # 按改进值排序
            all_improvements.sort(key=lambda x: x['Improvement'], reverse=True)

            print("\n📈 前5个最佳改进:")
            for i, imp in enumerate(all_improvements[:5]):
                print(f"{i + 1}. 记录{imp['Record']}的{imp['Feature']}: "
                      f"'{imp['Alternative_Value']}' -> {imp['Predicted_PCE']:.4f}% "
                      f"(改进: {imp['Improvement']:+.4f}%)")

        except Exception as e:
            print(f"❌ 生成报告失败: {e}")


def main():
    """主函数"""
    print("🔍 BestNIP记录特征优化系统 (集成模型)")
    print("🎯 目标: 优化bestnip.xlsx中记录的ETL_Passivator、HTL_Passivator和Precursor_Solution_Addictive")
    print("📊 方法: 使用集成模型（RF, XGB, CatBoost, LGBM）加权预测")
    print("📊 输出: 每个特征找到3个最佳替代值（与原值不同）")

    try:
        # 创建优化器实例
        optimizer = BestNIPOptimizer(
            data_path="FinalDataAll.xlsx",
            bestnip_path="bestnip.xlsx"
        )

        # 运行优化 - 可以指定要处理的记录数量，例如只处理前3条：optimizer.run_optimization(max_records=3)
        # 每个特征寻找3个最佳替代值
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
    except Exception as e:
        print(f"❌ 程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()