import pandas as pd
import os

class PCEAnalyzer:
    def __init__(self, data_path="FinalData10012.xlsx", mapping_path="label_mappings/full_mapping_summary.csv"):
        self.data_path = data_path
        self.mapping_path = mapping_path
        # 定义筛选条件：Structure = p-i-n
        self.target_conditions = {
            'Structure': 'p-i-n'
        }
        # 定义数值型字段的非零筛选条件 (保持原始列名)
        self.numeric_non_zero_conditions = [
            'total_scribing_line_width(μm)',
            'P1Width(μm)',
            'P2Width(μm)',
            'P3Width(μm)'
        ]
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
        # 1. 构建分类特征查询条件（使用编码值）
        query_parts = []
        for feature, target_value in self.target_conditions.items():
            encoded_val = self.all_mappings[feature]['original_to_encoded'].get(target_value)
            if encoded_val is None:
                raise ValueError(f"字段 {feature} 没有 '{target_value}' 的映射值")
            # 使用反引号包裹特征名，以防将来特征名也包含特殊字符
            query_part = f"`{feature}` == {encoded_val}"
            query_parts.append(query_part)

        # 2. 构建数值型非零查询条件 (使用反引号包裹列名)
        for feature in self.numeric_non_zero_conditions:
            if feature in self.df.columns:
                # 使用反引号 (`) 包裹包含特殊字符的列名
                # 使用 > 0 来确保值大于零且不是 NaN
                query_part = f"`{feature}` > 0"
                query_parts.append(query_part)
            else:
                print(f"警告: 字段 '{feature}' 在数据中不存在。")

        # 3. 执行查询
        if query_parts:
            query_str = " & ".join(query_parts)
            # print(f"Debug: Query String -> {query_str}") # 可用于调试
            matched = self.df.query(query_str)
        else:
            # 如果没有任何条件，使用全部数据（理论上不会发生）
            matched = self.df

        # 4. 筛选Active_Area并取Top5 (修改面积范围为10-22)
        result = matched[
            (matched['Active_Area'] >= 10) &
            (matched['Active_Area'] < 22)
        ].nlargest(5, 'PCE').copy()

        return result

    def print_full_results(self, result_df):
        """打印完整结果（所有字段的原始值）"""
        if result_df.empty:
            print("未找到匹配记录")
            return

        print(f"\n=== 匹配条件 ===")
        # 打印分类特征筛选条件
        condition_strs = []
        for k, v in self.target_conditions.items():
            if k in self.all_mappings:
                encoded_val = self.all_mappings[k]['original_to_encoded'].get(v)
                if encoded_val is not None:
                    condition_strs.append(f"{k}={v} (编码:{encoded_val})")
                else:
                    condition_strs.append(f"{k}={v} (?)")
            else:
                condition_strs.append(f"{k}={v}")
        # 打印数值型非零筛选条件 (显示原始列名即可)
        numeric_conditions_str = " & ".join([f"{f} > 0" for f in self.numeric_non_zero_conditions])
        condition_strs.append(numeric_conditions_str)

        print(" | ".join(condition_strs))
        print(f"Active_Area范围: [10, 22)")
        print(f"找到 {len(result_df)} 条匹配记录，按PCE降序:\n")

        for idx, row in result_df.iterrows():
            print(f"🔷 记录 (索引: {idx}) (PCE: {row['PCE']:.2f}%, Active_Area: {row['Active_Area']:.2f})")

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
            # 为输出DataFrame添加原始值列
            output_df = top_records.copy()
            for col in output_df.columns:
                if col in analyzer.all_mappings:
                    output_df[f"{col}_原始值"] = output_df[col].apply(
                        lambda x: analyzer.decode_feature(col, x)
                    )
            # 更新输出文件名以反映筛选条件
            output_file_name = "pce_predict/top_matches_Structure_p-i-n_ActiveArea_10-22_ScribingNonZero.xlsx"
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
            output_df.to_excel(output_file_name, index=False)
            print(f"\n结果已保存到 {output_file_name}")
    except Exception as e:
        print(f"错误: {str(e)}")




