import pandas as pd
import os

class PCEAnalyzer:
    def __init__(self, data_path="2FinalData0721.xlsx", mapping_path="label_mappings/full_mapping_summary.csv"):
        self.data_path = data_path
        self.mapping_path = mapping_path
        self.target_conditions = {
            'HTL': 'NiOx',
            'ETL': 'C60',
            # 'ETL-2':'BCP',
            'Metal_Electrode': ['Ag', 'Au', 'Cu'],
            'Glass': ['ITO', 'FTO']
        }
        self.df = None
        self.all_mappings = {}  # 存储所有字段的映射关系

        self.load_data()
        self.load_all_mappings()  # 加载所有字段的映射

    def load_data(self):
        """加载数据（确保所有特征已是数值型）"""
        self.df = pd.read_excel(self.data_path)
        print(f"数据加载完成，总记录数: {len(self.df)}")
        print("包含字段:", list(self.df.columns))

    def load_all_mappings(self):
        """加载所有字段的映射关系（不只是目标特征）"""
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"映射文件不存在: {self.mapping_path}")

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
        print("全字段映射加载完成")

    def decode_feature(self, feature_name, encoded_value):
        """将编码值转换回原始值"""
        if feature_name in self.all_mappings:
            return self.all_mappings[feature_name]['encoded_to_original'].get(encoded_value, str(encoded_value))
        return encoded_value  # 如果字段没有映射关系，直接返回值

    def find_top_records(self):
        """查找符合条件的前5条记录"""
        # 构建查询条件（使用编码值）
        query_parts = []
        for feature, targets in self.target_conditions.items():
            if isinstance(targets, list):
                # 多个目标值的情况
                encoded_vals = [self.all_mappings[feature]['original_to_encoded'].get(target) for target in targets]
                if any(val is None for val in encoded_vals):
                    raise ValueError(f"字段 {feature} 没有某些目标值的映射值")
                query_part = f"{feature}.isin({encoded_vals})"
            else:
                # 单个目标值的情况
                encoded_val = self.all_mappings[feature]['original_to_encoded'].get(targets)
                if encoded_val is None:
                    raise ValueError(f"字段 {feature} 没有 {targets} 的映射值")
                query_part = f"{feature} == {encoded_val}"
            query_parts.append(query_part)

        # 执行查询
        query_str = " & ".join(query_parts)
        matched = self.df.query(query_str)

        # 筛选Active_Area并取Top5
        result = matched[
            (matched['Active_Area'] >= 10) &
            (matched['Active_Area'] < 20)
            ].nlargest(5, 'PCE').copy()

        return result

    def print_full_results(self, result_df):
        """打印完整结果（所有字段的原始值）"""
        if result_df.empty:
            print("未找到匹配记录")
            return

        print(f"\n=== 匹配条件 ===")
        print(" | ".join([f"{k}={v}" for k, v in self.target_conditions.items()]))
        print(f"Active_Area范围: [10, 20)")
        print(f"找到 {len(result_df)} 条匹配记录，按PCE降序:\n")

        for idx, row in result_df.iterrows():
            print(f"🔷 记录 {idx} (PCE: {row['PCE']:.2f}%, Active_Area: {row['Active_Area']:.2f})")

            # 打印所有字段（按原始值显示）
            for col in result_df.columns:
                if col in self.all_mappings:  # 有映射关系的字段
                    original_val = self.decode_feature(col, row[col])
                    print(f"  {col}: {original_val} (编码值: {row[col]})")
                else:  # 数值型字段
                    print(f"  {col}: {row[col]}")

            print("─" * 50)


if __name__ == "__main__":
    try:
        analyzer = PCEAnalyzer()
        top_records = analyzer.find_top_records()
        analyzer.print_full_results(top_records)

        if not top_records.empty:
            output_df = top_records.copy()
            for col in output_df.columns:
                if col in analyzer.all_mappings:
                    output_df[f"{col}_原始值"] = output_df[col].apply(
                        lambda x: analyzer.decode_feature(col, x)
                    )
            output_df.to_excel("pce_predict/top_matches_with_original_values[10-20)pin.xlsx", index=False)
            print("\n结果已保存到 pce_predict/top_matches_with_original_values[10-20)pin.xlsx")
    except Exception as e:
        print(f"错误: {str(e)}")



