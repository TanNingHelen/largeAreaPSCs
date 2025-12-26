import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class OffsetBasedOptimizer:
    def __init__(self, data_path="FinalData.xlsx",
                 bestnip_path="bestnip.xlsx",
                 min_improvement=0.1,  # 最小改进值
                 max_improvement=1.0):  # 最大改进值
        self.data_path = data_path
        self.bestnip_path = bestnip_path
        self.min_improvement = min_improvement
        self.max_improvement = max_improvement

        # 要优化的特征
        self.target_features = [
            'ETL_Passivator',
            'HTL_Passivator',
            'Precursor_Solution_Addictive'
        ]

        # 加载数据
        self.df = None
        self.bestnip_records = None
        self.mapping_df = None
        self.original_mapping = {}  # 编码到原始值的映射

        # 缓存特征分析结果
        self.feature_impact_cache = {}

        # 结果存储
        self.optimization_results = {}

        # 创建结果目录
        self.results_dir = 'offset_optimization_results'
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

            # 显示记录的PCE值
            print(f"\n📊 BestNIP记录PCE值:")
            for idx, row in self.bestnip_records.iterrows():
                record_id = row.get('Record', f'Record_{idx + 1}')
                pce = row.get('PCE', 'N/A')
                print(f"  记录 {idx + 1} (ID: {record_id}): PCE = {pce}%")

        except Exception as e:
            print(f"❌ 加载BestNIP记录失败: {e}")
            raise

        # 加载数据库
        try:
            self.df = pd.read_excel(self.data_path)
            print(f"✅ 数据库加载成功: {len(self.df)} 条记录")

        except Exception as e:
            print(f"❌ 加载数据库失败: {e}")
            raise

        # 加载映射文件
        self.load_mapping_file()

        # 分析数据库中的PCE分布
        self.analyze_pce_distribution()

    def load_mapping_file(self):
        """加载映射文件，构建双向映射"""
        mapping_paths = [
            'label_mappings/full_mapping_summary.csv',
            '../label_mappings/full_mapping_summary.csv',
            './label_mappings/full_mapping_summary.csv'
        ]

        for path in mapping_paths:
            if os.path.exists(path):
                try:
                    self.mapping_df = pd.read_csv(path)
                    print(f"✅ 映射文件加载成功: {path}")

                    # 构建编码到原始值的映射
                    self.original_mapping = {}
                    for feature in self.target_features:
                        feature_mapping = self.mapping_df[self.mapping_df['Feature'] == feature]
                        if len(feature_mapping) > 0:
                            encoded_to_original = {}
                            for _, row in feature_mapping.iterrows():
                                encoded_value = row['Encoded']
                                original_value = row['Original']
                                if pd.isna(original_value):
                                    original_value = ''
                                encoded_to_original[encoded_value] = original_value

                            self.original_mapping[feature] = encoded_to_original
                            print(f"  特征 '{feature}' 有 {len(encoded_to_original)} 个映射")

                    return

                except Exception as e:
                    print(f"❌ 加载映射文件失败 {path}: {e}")

        print("⚠️ 未找到映射文件")

    def analyze_pce_distribution(self):
        """分析数据库中的PCE分布"""
        print(f"\n📊 数据库PCE分析:")
        print(f"  平均值: {self.df['PCE'].mean():.2f}%")
        print(f"  中位数: {self.df['PCE'].median():.2f}%")
        print(f"  最大值: {self.df['PCE'].max():.2f}%")
        print(f"  最小值: {self.df['PCE'].min():.2f}%")
        print(f"  标准差: {self.df['PCE'].std():.2f}%")

        # 识别高PCE数据（前20%）
        high_pce_threshold = self.df['PCE'].quantile(0.8)
        high_pce_df = self.df[self.df['PCE'] >= high_pce_threshold]
        print(f"  高PCE阈值（前20%）: {high_pce_threshold:.2f}%")
        print(f"  高PCE记录数: {len(high_pce_df)}")

    def decode_value(self, feature_name, encoded_value):
        """将编码值解码为原始值"""
        if feature_name in self.original_mapping:
            # 处理空值
            if pd.isna(encoded_value) or encoded_value == '':
                return ''

            # 尝试转换为整数
            try:
                if isinstance(encoded_value, str):
                    encoded_int = int(float(encoded_value))
                else:
                    encoded_int = int(encoded_value)

                original_value = self.original_mapping[feature_name].get(encoded_int, str(encoded_value))
                return original_value
            except:
                return str(encoded_value)
        return str(encoded_value)

    def analyze_feature_impact(self, feature_name):
        """分析特征取值对PCE的影响，使用缓存提高效率"""
        if feature_name in self.feature_impact_cache:
            return self.feature_impact_cache[feature_name]

        if feature_name not in self.df.columns:
            return {}

        # 筛选掉空值
        valid_data = self.df[self.df[feature_name].notna()].copy()

        if len(valid_data) == 0:
            return {}

        # 对每个取值计算统计信息
        impact_results = {}

        # 计算总体平均PCE
        overall_mean = valid_data['PCE'].mean()

        for value in valid_data[feature_name].unique():
            # 筛选该取值的记录
            value_data = valid_data[valid_data[feature_name] == value]

            if len(value_data) > 0:
                # 计算统计信息
                pce_mean = value_data['PCE'].mean()
                pce_std = value_data['PCE'].std()
                count = len(value_data)

                # 计算与总体平均的相对表现
                relative_performance = pce_mean - overall_mean

                # 解码原始值
                original_value = self.decode_value(feature_name, value)

                impact_results[value] = {
                    'original_value': original_value,
                    'encoded_value': value,
                    'pce_mean': pce_mean,
                    'pce_std': pce_std,
                    'count': count,
                    'relative_performance': relative_performance,
                    'display': f"{original_value} ({value})"
                }

        # 缓存结果
        self.feature_impact_cache[feature_name] = impact_results

        return impact_results

    def calculate_offset_predictions(self, original_pce, feature_impact, feature_name, original_encoded, record_id):
        """基于偏移计算预测PCE，考虑原始记录的PCE水平"""
        predictions = []

        # 获取当前特征的总体信息
        overall_mean = self.df['PCE'].mean()
        overall_std = self.df['PCE'].std()

        for encoded_value, impact_info in feature_impact.items():
            # 跳过原始值本身
            if str(encoded_value) == str(original_encoded):
                continue

            # 获取该取值的平均PCE
            value_mean = impact_info['pce_mean']
            value_std = impact_info['pce_std']
            count = impact_info['count']

            # 计算改进值 - 基于原始记录的PCE水平
            # 方法1：如果原始PCE低于该取值平均PCE，则预测为该取值平均PCE
            # 方法2：基于原始PCE与取值平均PCE的差距
            improvement = value_mean - original_pce

            # 调整改进值，避免过大或过小
            if improvement < self.min_improvement:
                # 如果改进值太小，使用相对性能来调整
                relative_improvement = self.min_improvement * (1 + impact_info['relative_performance'] / 10)
                improvement = max(self.min_improvement, min(relative_improvement, self.max_improvement))
            elif improvement > self.max_improvement:
                improvement = self.max_improvement

            # 确保改进值在范围内
            improvement = max(self.min_improvement, min(improvement, self.max_improvement))

            # 预测PCE = 原始PCE + 改进值
            predicted_pce = original_pce + improvement

            # 添加置信度（基于数据量和标准差）
            # 数据越多，置信度越高；标准差越小，置信度越高
            base_confidence = min(80, count * 3)  # 每1个数据点增加3%置信度，最高80%

            # 基于标准差调整置信度
            if value_std > 0:
                std_factor = max(0, 1 - (value_std / overall_std) * 0.5)
                confidence = base_confidence * std_factor
            else:
                confidence = base_confidence

            confidence = min(100, max(20, confidence))  # 确保在20-100%之间

            predictions.append({
                'record_id': record_id,
                'encoded_value': encoded_value,
                'original_value': impact_info['original_value'],
                'display_value': impact_info['display'],
                'predicted_pce': round(predicted_pce, 4),
                'original_pce': original_pce,
                'value_mean_pce': round(value_mean, 4),
                'improvement': round(improvement, 4),
                'value_pce_std': round(value_std, 4),
                'data_count': count,
                'confidence': round(confidence, 1),
                'relative_performance': round(impact_info['relative_performance'], 4),
                'method': 'value_mean_based' if improvement >= self.min_improvement else 'relative_performance_based'
            })

        return predictions

    def optimize_feature_for_record(self, record_idx, record):
        """为单条记录优化特征"""
        record_id = record.get('Record', f'Record_{record_idx + 1}')
        original_pce = record.get('PCE', 0)

        print(f"\n{'=' * 60}")
        print(f"🚀 优化记录 {record_idx + 1} (ID: {record_id}, PCE: {original_pce:.2f}%)")
        print(f"{'=' * 60}")

        record_results = {}

        # 对每个特征进行优化
        for feature_idx, feature_name in enumerate(self.target_features, 1):
            print(f"\n🔍 优化特征 {feature_idx}/3: {feature_name}")

            # 获取原始值
            original_encoded = record.get(feature_name, '')
            original_decoded = self.decode_value(feature_name, original_encoded)
            print(f"  原始值: '{original_decoded}' (编码: {original_encoded})")

            # 分析该特征的影响
            feature_impact = self.analyze_feature_impact(feature_name)

            if not feature_impact:
                print(f"  ⚠️ 无法分析该特征的影响")
                continue

            # 显示原始值的表现（如果存在）
            if original_encoded in feature_impact:
                original_impact = feature_impact[original_encoded]
                print(f"  原始值的平均PCE: {original_impact['pce_mean']:.4f}% (基于{original_impact['count']}条数据)")
                print(f"  原始值相对性能: {original_impact['relative_performance']:.4f}")

            # 计算基于偏移的预测
            predictions = self.calculate_offset_predictions(
                original_pce, feature_impact, feature_name, original_encoded, record_id
            )

            if not predictions:
                print(f"  ⚠️ 没有找到有效的替代取值")
                continue

            # 按预测PCE降序排序
            predictions.sort(key=lambda x: x['predicted_pce'], reverse=True)

            # 取前3个
            top_3 = predictions[:3]

            print(f"\n  📊 前3个最佳取值:")
            for i, pred in enumerate(top_3):
                print(f"    {i + 1}. {pred['display_value']}")
                print(f"        预测PCE: {pred['predicted_pce']:.4f}%, "
                      f"改进: {pred['improvement']:+.4f}%")
                print(f"        该取值平均PCE: {pred['value_mean_pce']:.4f}%, "
                      f"相对性能: {pred['relative_performance']:+.4f}")
                print(f"        数据量: {pred['data_count']}, 置信度: {pred['confidence']:.1f}%")
                print(f"        计算方法: {pred['method']}")

            # 存储结果
            record_results[feature_name] = {
                'original_encoded': original_encoded,
                'original_decoded': original_decoded,
                'original_pce': original_pce,
                'top_3_values': top_3
            }

        return {
            'record_id': record_id,
            'original_pce': original_pce,
            'optimization_results': record_results
        }

    def run_optimization(self):
        """运行优化过程"""
        print("\n" + "=" * 60)
        print("🎯 基于偏移的特征优化")
        print("=" * 60)
        print(f"📊 方法: 基于特征取值在数据库中的平均PCE表现计算偏移量")
        print(f"📈 预测PCE = 原始PCE + 基于取值平均PCE的改进值")
        print(f"📋 改进值范围: {self.min_improvement}% - {self.max_improvement}%")

        if self.bestnip_records is None or len(self.bestnip_records) == 0:
            print("❌ BestNIP文件中没有记录")
            return

        # 存储所有结果
        all_results = []

        # 对每条记录进行优化
        for record_idx, (_, record) in enumerate(self.bestnip_records.iterrows()):
            result = self.optimize_feature_for_record(record_idx, record)
            if result:
                all_results.append(result)

        # 保存结果
        self.save_results(all_results)

        # 生成报告
        self.generate_report(all_results)

        return all_results

    def save_results(self, all_results):
        """保存优化结果"""
        try:
            # 创建详细结果DataFrame
            detailed_results = []

            for result_idx, result in enumerate(all_results):
                record_id = result['record_id']
                original_pce = result['original_pce']

                for feature_name, feature_result in result['optimization_results'].items():
                    # 获取前3个值
                    for rank, top_value in enumerate(feature_result['top_3_values'], 1):
                        detailed_results.append({
                            'Record_Index': result_idx + 1,
                            'Record_ID': record_id,
                            'Original_PCE': original_pce,
                            'Feature': feature_name,
                            'Rank': rank,
                            'Original_Value_Encoded': feature_result['original_encoded'],
                            'Original_Value_Decoded': feature_result['original_decoded'],
                            'Recommended_Value_Encoded': top_value['encoded_value'],
                            'Recommended_Value_Decoded': top_value['original_value'],
                            'Display_Value': top_value['display_value'],
                            'Predicted_PCE': top_value['predicted_pce'],
                            'Improvement': top_value['improvement'],
                            'Value_Mean_PCE': top_value['value_mean_pce'],
                            'Value_PCE_Std': top_value['value_pce_std'],
                            'Relative_Performance': top_value['relative_performance'],
                            'Data_Count': top_value['data_count'],
                            'Confidence': top_value['confidence'],
                            'Calculation_Method': top_value['method']
                        })

            # 转换为DataFrame
            results_df = pd.DataFrame(detailed_results)

            # 保存到Excel
            filename = f"{self.results_dir}/offset_optimization_top3_{self.timestamp}.xlsx"
            results_df.to_excel(filename, index=False)
            print(f"\n💾 详细结果已保存: {filename}")

            # 同时保存为CSV格式
            csv_filename = f"{self.results_dir}/offset_optimization_top3_{self.timestamp}.csv"
            results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"💾 结果已保存为CSV: {csv_filename}")

            # 保存汇总结果
            self.save_summary_results(all_results)

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

    def save_summary_results(self, all_results):
        """保存汇总结果"""
        try:
            # 创建汇总表格
            summary_data = []

            for result_idx, result in enumerate(all_results):
                record_id = result['record_id']
                original_pce = result['original_pce']

                summary_row = {
                    'Record_Index': result_idx + 1,
                    'Record_ID': record_id,
                    'Original_PCE': original_pce
                }

                # 添加每个特征的最佳推荐（第1名）
                for feature_name in self.target_features:
                    if feature_name in result['optimization_results']:
                        top_values = result['optimization_results'][feature_name]['top_3_values']
                        if top_values:
                            best_value = top_values[0]
                            summary_row[f'{feature_name}_Original'] = result['optimization_results'][feature_name][
                                'original_decoded']
                            summary_row[f'{feature_name}_Best_Value'] = best_value['original_value']
                            summary_row[f'{feature_name}_Display'] = best_value['display_value']
                            summary_row[f'{feature_name}_Predicted_PCE'] = best_value['predicted_pce']
                            summary_row[f'{feature_name}_Improvement'] = best_value['improvement']
                            summary_row[f'{feature_name}_Confidence'] = best_value['confidence']
                            summary_row[f'{feature_name}_Method'] = best_value['method']

                summary_data.append(summary_row)

            # 创建汇总DataFrame
            summary_df = pd.DataFrame(summary_data)

            # 保存汇总结果
            summary_filename = f"{self.results_dir}/offset_summary_{self.timestamp}.xlsx"
            summary_df.to_excel(summary_filename, index=False)
            print(f"💾 汇总结果已保存: {summary_filename}")

        except Exception as e:
            print(f"❌ 保存汇总结果失败: {e}")

    def generate_report(self, all_results):
        """生成优化报告"""
        try:
            # 计算总体统计
            overall_mean_pce = self.df['PCE'].mean()

            report_content = []
            report_content.append("基于偏移的特征优化报告")
            report_content.append("=" * 70)
            report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"数据库文件: {self.data_path}")
            report_content.append(f"目标记录文件: {self.bestnip_path}")
            report_content.append(f"优化特征: {', '.join(self.target_features)}")
            report_content.append(f"改进值范围: {self.min_improvement}% - {self.max_improvement}%")
            report_content.append(f"数据库总体平均PCE: {overall_mean_pce:.4f}%")
            report_content.append("")

            report_content.append("📊 优化方法说明:")
            report_content.append("1. 分析数据库中每个特征取值的平均PCE表现")
            report_content.append("2. 对于每条记录，基于原始PCE和特征取值的平均PCE计算改进值")
            report_content.append("3. 预测PCE = 原始PCE + 基于取值平均PCE的改进值")
            report_content.append("4. 置信度基于数据量和标准差计算")
            report_content.append("5. 选择预测PCE最高的前3个取值")
            report_content.append("")

            report_content.append("📊 优化结果:")
            report_content.append("-" * 70)

            for result_idx, result in enumerate(all_results):
                record_id = result['record_id']
                original_pce = result['original_pce']

                report_content.append(f"\n记录 {result_idx + 1} (ID: {record_id}):")
                report_content.append(f"  原始PCE: {original_pce:.2f}%")

                for feature_name in self.target_features:
                    if feature_name in result['optimization_results']:
                        feature_result = result['optimization_results'][feature_name]

                        report_content.append(f"\n  {feature_name}:")
                        report_content.append(f"    原始值: {feature_result['original_decoded']}")

                        for i, top_value in enumerate(feature_result['top_3_values'], 1):
                            report_content.append(f"    第{i}名: {top_value['display_value']}")
                            report_content.append(f"        预测PCE: {top_value['predicted_pce']:.4f}%, "
                                                  f"改进: {top_value['improvement']:+.4f}%")
                            report_content.append(f"        该取值平均PCE: {top_value['value_mean_pce']:.4f}%, "
                                                  f"相对性能: {top_value['relative_performance']:+.4f}")
                            report_content.append(f"        置信度: {top_value['confidence']:.1f}%, "
                                                  f"数据量: {top_value['data_count']}条")
                            report_content.append(f"        计算方法: {top_value['method']}")

                report_content.append("-" * 40)

            # 保存报告
            report_filename = f"{self.results_dir}/offset_optimization_report_{self.timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))

            print(f"📋 优化报告已保存: {report_filename}")

            # 打印报告摘要
            print("\n" + "=" * 60)
            print("📊 优化完成!")
            print("=" * 60)
            print(f"✅ 已优化 {len(all_results)} 条记录")
            print(f"✅ 每个特征找到前3个最佳取值")
            print(f"✅ 基于取值平均PCE计算方法，改进值限制在{self.min_improvement}-{self.max_improvement}之间")
            print(f"✅ 结果保存在: {self.results_dir}/ 目录下")

            # 显示最佳改进
            print(f"\n🏆 最佳改进总结:")

            best_improvements = []
            for result_idx, result in enumerate(all_results):
                for feature_name in self.target_features:
                    if feature_name in result['optimization_results']:
                        top_values = result['optimization_results'][feature_name]['top_3_values']
                        if top_values:
                            best_value = top_values[0]
                            best_improvements.append({
                                'Record': result_idx + 1,
                                'Feature': feature_name,
                                'Best_Value': best_value['original_value'],
                                'Display': best_value['display_value'],
                                'Predicted_PCE': best_value['predicted_pce'],
                                'Improvement': best_value['improvement'],
                                'Confidence': best_value['confidence'],
                                'Method': best_value['method']
                            })

            # 按预测PCE排序
            best_improvements.sort(key=lambda x: x['Predicted_PCE'], reverse=True)

            for i, imp in enumerate(best_improvements[:5]):
                print(f"{i + 1}. 记录{imp['Record']}的{imp['Feature']}: {imp['Display']}")
                print(f"   预测PCE: {imp['Predicted_PCE']:.4f}%, "
                      f"改进: {imp['Improvement']:+.4f}%, "
                      f"置信度: {imp['Confidence']:.1f}%, "
                      f"方法: {imp['Method']}")

        except Exception as e:
            print(f"❌ 生成报告失败: {e}")


def main():
    """主函数"""
    print("🔍 基于偏移的特征优化系统")
    print("🎯 目标: 优化bestnip.xlsx中记录的ETL_Passivator、HTL_Passivator和Precursor_Solution_Addictive")
    print("📊 输出: 每个特征替换后PCE最高的前3个取值")
    print(f"🔄 方法: 基于特征取值在数据库中的平均PCE表现计算改进值，改进值限制在0.1-1.0之间")
    print(f"📁 数据集: FinalData.xlsx")

    try:
        # 创建优化器实例
        optimizer = OffsetBasedOptimizer(
            data_path="FinalData.xlsx",
            bestnip_path="bestnip.xlsx",
            min_improvement=0.1,  # 最小改进值0.1%
            max_improvement=1.0  # 最大改进值1.0%
        )

        # 运行优化
        results = optimizer.run_optimization()

    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("💡 请确保以下文件存在:")
        print("   - FinalData.xlsx (数据库文件)")
        print("   - bestnip.xlsx (目标记录文件)")
    except Exception as e:
        print(f"❌ 程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()