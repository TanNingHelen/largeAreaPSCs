import pandas as pd
import os


class PCEAnalyzer:
    def __init__(self, data_path="FinalData10312All.xlsx", mapping_path="label_mappings/full_mapping_summary.csv"):
        self.data_path = data_path
        self.mapping_path = mapping_path
        # 不再设置特定的目标条件，寻找所有结构中的高效数据
        self.df = None
        self.all_mappings = {}  # 存储所有字段的映射

        self.load_data()
        self.load_all_mappings()  # 加载所有字段的映射

    def load_data(self):
        """加载数据（确保所有特征都是数值类型）"""
        try:
            self.df = pd.read_excel(self.data_path)
            print(f"✅ 数据加载成功，总记录数: {len(self.df)}")
            print("📊 数据字段:", list(self.df.columns))

            # 检查Active_Area列是否存在
            if 'Active_Area' not in self.df.columns:
                print("⚠️ 警告: 数据集中没有找到Active_Area字段")

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise

    def load_all_mappings(self):
        """加载所有字段的映射（不仅仅是目标特征）"""
        if not os.path.exists(self.mapping_path):
            print(f"⚠️ 映射文件未找到: {self.mapping_path}")
            print("⚠️ 将直接使用编码值显示数据")
            return

        try:
            mapping_df = pd.read_csv(self.mapping_path)
            for feature in mapping_df['Feature'].unique():
                self.all_mappings[feature] = {
                    'original_to_encoded': {},
                    'encoded_to_original': {}
                }
                feature_data = mapping_df[mapping_df['Feature'] == feature]

                for _, row in feature_data.iterrows():
                    # 双向映射
                    self.all_mappings[feature]['original_to_encoded'][row['Original']] = row['Encoded']
                    self.all_mappings[feature]['encoded_to_original'][row['Encoded']] = row['Original']
            print("✅ 所有字段映射加载成功")
        except Exception as e:
            print(f"❌ 映射文件加载失败: {e}")

    def decode_feature(self, feature_name, encoded_value):
        """将编码值转换回原始值"""
        if feature_name in self.all_mappings:
            # 处理NaN值
            if pd.isna(encoded_value):
                return "N/A"
            # 确保encoded_value是整数类型
            try:
                encoded_int = int(encoded_value)
                return self.all_mappings[feature_name]['encoded_to_original'].get(encoded_int, str(encoded_value))
            except (ValueError, TypeError):
                return str(encoded_value)
        return encoded_value  # 如果没有映射，直接返回值

    def find_top_records(self, n=20):
        """寻找面积在10-25平方厘米的高效数据"""
        # 检查Active_Area列是否存在
        if 'Active_Area' not in self.df.columns:
            print("❌ 错误: 数据集中没有Active_Area字段")
            return pd.DataFrame()

        # 检查PCE列是否存在
        if 'PCE' not in self.df.columns:
            print("❌ 错误: 数据集中没有PCE字段")
            return pd.DataFrame()

        # 筛选面积在10-25平方厘米的记录
        area_filtered = self.df[
            (self.df['Active_Area'] >= 10) &
            (self.df['Active_Area'] <= 25)
            ]

        print(f"📊 面积在10-25平方厘米的记录数: {len(area_filtered)}")

        if len(area_filtered) == 0:
            print("⚠️ 没有找到面积在10-25平方厘米的记录")
            # 显示面积的范围，帮助用户了解数据
            if 'Active_Area' in self.df.columns:
                print(
                    f"📏 数据集中Active_Area的范围: {self.df['Active_Area'].min():.2f} - {self.df['Active_Area'].max():.2f}")
            return pd.DataFrame()

        # 按PCE降序排列，取前n个
        result = area_filtered.nlargest(n, 'PCE').copy()

        return result

    def print_full_results(self, result_df):
        """打印完整结果（所有字段的原始值）"""
        if result_df.empty:
            print("❌ 没有找到匹配的记录")
            return

        print(f"\n{'=' * 60}")
        print(f"🔍 搜索结果: 面积在10-25平方厘米的高效数据")
        print(f"📏 Active_Area范围: [10, 25] 平方厘米")
        print(f"📈 找到 {len(result_df)} 条记录，按PCE降序排列:\n")

        for i, (idx, row) in enumerate(result_df.iterrows(), 1):
            print(f"🏆 第{i}名 (PCE: {row['PCE']:.2f}%, 面积: {row['Active_Area']:.2f} cm²)")

            # 定义需要显示的重要字段（按优先级排序）
            important_fields = [
                'Record', 'PCE', 'Active_Area',
                'Structure', 'HTL', 'ETL', 'Perovskite',
                'Jsc', 'Voc', 'FF', 'PCE_std',
                'HTL-2', 'ETL-2', 'Metal_Electrode', 'Glass',
                'Precursor_Solution', 'Deposition_Method',
                'total_scribing_line_width(μm)', 'P1Width(μm)', 'P2Width(μm)', 'P3Width(μm)',
                'P1_P2Scribing_Spacing(μm)', 'P2_P3Scribing_Spacing(μm)',
                'GFF', 'submodule_number', 'Type'
            ]

            # 显示重要字段
            for col in important_fields:
                if col in result_df.columns:
                    if col in self.all_mappings:  # 有映射的字段
                        original_val = self.decode_feature(col, row[col])
                        print(f"  {col}: {original_val}")
                    else:  # 数值字段
                        val = row[col]
                        if pd.isna(val):
                            print(f"  {col}: N/A")
                        else:
                            print(f"  {col}: {val}")

            print(f"{'-' * 60}")

    def analyze_results(self, result_df):
        """分析结果数据的统计信息"""
        if result_df.empty:
            return

        print(f"\n📊 搜索结果统计信息:")
        print(f"{'=' * 60}")

        # 基本统计
        print(f"📈 PCE统计:")
        print(f"  - 平均值: {result_df['PCE'].mean():.2f}%")
        print(f"  - 最大值: {result_df['PCE'].max():.2f}%")
        print(f"  - 最小值: {result_df['PCE'].min():.2f}%")
        print(f"  - 标准差: {result_df['PCE'].std():.2f}%")

        # 面积统计
        print(f"\n📏 面积统计:")
        print(f"  - 平均面积: {result_df['Active_Area'].mean():.2f} cm²")
        print(f"  - 面积范围: {result_df['Active_Area'].min():.2f} - {result_df['Active_Area'].max():.2f} cm²")

        # 结构分布（如果存在Structure字段）
        if 'Structure' in result_df.columns:
            print(f"\n🏗️ 结构类型分布:")
            if 'Structure' in self.all_mappings:
                # 解码结构类型
                structures = result_df['Structure'].apply(lambda x: self.decode_feature('Structure', x))
            else:
                structures = result_df['Structure']

            structure_counts = structures.value_counts()
            for structure, count in structure_counts.items():
                percentage = (count / len(result_df)) * 100
                print(f"  - {structure}: {count} 条 ({percentage:.1f}%)")

        # HTL分布（如果存在HTL字段）
        if 'HTL' in result_df.columns and 'HTL' in self.all_mappings:
            print(f"\n🔄 HTL材料分布:")
            htl_types = result_df['HTL'].apply(lambda x: self.decode_feature('HTL', x))
            htl_counts = htl_types.value_counts()
            for htl, count in htl_counts.head(5).items():  # 显示前5种
                percentage = (count / len(result_df)) * 100
                print(f"  - {htl}: {count} 条 ({percentage:.1f}%)")

        # ETL分布（如果存在ETL字段）
        if 'ETL' in result_df.columns and 'ETL' in self.all_mappings:
            print(f"\n🔄 ETL材料分布:")
            etl_types = result_df['ETL'].apply(lambda x: self.decode_feature('ETL', x))
            etl_counts = etl_types.value_counts()
            for etl, count in etl_counts.head(5).items():  # 显示前5种
                percentage = (count / len(result_df)) * 100
                print(f"  - {etl}: {count} 条 ({percentage:.1f}%)")


if __name__ == "__main__":
    try:
        print("🔍 开始搜索面积在10-25平方厘米的高效钙钛矿太阳能电池数据...")
        analyzer = PCEAnalyzer(data_path="FinalData10312All.xlsx")

        # 寻找前20个高效记录
        top_records = analyzer.find_top_records(n=20)

        if not top_records.empty:
            # 打印结果
            analyzer.print_full_results(top_records)

            # 分析结果统计信息
            analyzer.analyze_results(top_records)

            # 保存结果到Excel文件
            output_df = top_records.copy()

            # 为有映射的字段添加原始值列
            for col in output_df.columns:
                if col in analyzer.all_mappings:
                    # 创建新列，包含原始值
                    new_col_name = f"{col}_原始值"
                    output_df[new_col_name] = output_df[col].apply(
                        lambda x: analyzer.decode_feature(col, x)
                    )

            # 确保输出目录存在
            output_dir = "pce_predict"
            os.makedirs(output_dir, exist_ok=True)

            # 生成输出文件名
            output_filename = os.path.join(output_dir, "top_high_pce_records_area_10-25.xlsx")
            output_df.to_excel(output_filename, index=False)
            print(f"\n💾 结果已保存到: {output_filename}")

            # 创建简化的结果文件（只包含重要字段）
            important_columns = [
                'Record', 'PCE', 'Active_Area', 'Structure', 'HTL', 'ETL',
                'Perovskite', 'Jsc', 'Voc', 'FF', 'GFF', 'total_scribing_line_width(μm)'
            ]

            # 只保留实际存在的列
            available_columns = [col for col in important_columns if col in output_df.columns]
            simplified_df = output_df[available_columns].copy()

            # 为简化文件中的字段添加原始值
            for col in ['Structure', 'HTL', 'ETL']:
                if col in simplified_df.columns and col in analyzer.all_mappings:
                    original_col = f"{col}_原始值"
                    if original_col in output_df.columns:
                        simplified_df[original_col] = output_df[original_col]

            simplified_filename = os.path.join(output_dir, "simplified_top_records.xlsx")
            simplified_df.to_excel(simplified_filename, index=False)
            print(f"💾 简化结果已保存到: {simplified_filename}")

        else:
            print("❌ 没有找到符合条件的记录")

    except FileNotFoundError as e:
        print(f"❌ 文件未找到错误: {e}")
        print("💡 请确保以下文件存在:")
        print("   1. FinalData10312All.xlsx (主数据文件)")
        print("   2. label_mappings/full_mapping_summary.csv (映射文件，可选)")
    except Exception as e:
        print(f"❌ 程序执行过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()