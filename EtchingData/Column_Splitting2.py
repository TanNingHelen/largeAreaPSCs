import pandas as pd
import re
import os
from collections import defaultdict
#处理Perovskite列，创建新表格，新添加元素作为列名

# 修复了正则表达式警告并处理了Pb元素的特殊情况
def get_element_ratio(composition):
    elements = {'Cs': 'A', 'MA': 'A', 'FA': 'A', 'Rb': 'A', 'Pb': 'B', 'Sn': 'B', 'I': 'X', 'Br': 'X', 'Cl': 'X'}
    element_ratio = {key: 0 for key in elements.keys()}
    mul_factors = {key: [] for key in elements.keys()}

    # 处理带括号的部分
    if ')' in composition:
        pattern2 = r"$([^()]+)$"
        match = re.findall(pattern2, composition)
        for m in match:
            for ele in elements.keys():
                if ele in m:
                    temp_str = composition.split('(' + m + ')')[1]
                    pattern3 = r'([0-9]*\.?[0-9]*)'
                    mul_fact = re.findall(pattern3, temp_str)[0]
                    if not mul_fact:  # 处理空字符串
                        mul_fact = '1'
                    try:
                        mul_factors[ele].append(float(mul_fact))
                    except:
                        mul_factors[ele].append(1.0)  # 默认值为1

    # 处理所有元素
    for ele in elements.keys():
        if ele in composition:
            pattern3 = r'([0-9]*\.?[0-9]*)'
            if ele == 'Pb' and len(composition.split(ele)) > 2:
                element_ratio[ele] = []
                for temp_str in composition.split(ele)[1:3]:
                    if not re.match(r'\d', temp_str):  # 修复正则表达式警告
                        element_ratio[ele].append(1.0)
                    else:
                        num = re.findall(pattern3, temp_str)[0]
                        if not num:  # 处理空字符串
                            num = '1'
                        element_ratio[ele].append(float(num))
            else:
                parts = composition.split(ele)
                if len(parts) > 1:
                    temp_str = parts[1]
                    if not re.match(r'\d', temp_str):  # 修复正则表达式警告
                        element_ratio[ele] = 1.0
                    else:
                        num = re.findall(pattern3, temp_str)[0]
                        if not num:  # 处理空字符串
                            num = '1'
                        element_ratio[ele] = float(num)
                else:
                    element_ratio[ele] = 1.0

    # 修复了核心错误：处理Pb元素的列表情况
    for symbol, position in elements.items():
        if position == 'A':
            if mul_factors[symbol]:
                element_ratio[symbol] = mul_factors[symbol][0] * element_ratio[symbol]
        elif position == 'B':
            if symbol == 'Pb' and isinstance(element_ratio[symbol], list):
                # 特殊处理Pb元素的列表情况
                if len(mul_factors[symbol]) == 1:
                    element_ratio[symbol] = mul_factors[symbol][0] * sum(element_ratio[symbol])
                elif len(mul_factors[symbol]) == 2:
                    element_ratio[symbol] = mul_factors[symbol][0] * element_ratio[symbol][0] + mul_factors[symbol][1] * \
                                            element_ratio[symbol][1]
                else:
                    element_ratio[symbol] = sum(element_ratio[symbol])
            else:
                if mul_factors[symbol]:
                    if isinstance(element_ratio[symbol], list):
                        element_ratio[symbol] = mul_factors[symbol][0] * sum(element_ratio[symbol])
                    else:
                        element_ratio[symbol] = mul_factors[symbol][0] * element_ratio[symbol]
        else:  # X位元素
            if isinstance(element_ratio[symbol], list):
                element_ratio[symbol] = sum(element_ratio[symbol]) / 3.0
            else:
                element_ratio[symbol] /= 3.0

            if mul_factors[symbol]:
                element_ratio[symbol] = mul_factors[symbol][0] * element_ratio[symbol]

    return element_ratio


# 处理Excel文件
def process_perovskite_column(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取Excel文件: {file_path}")
        print(f"文件包含 {len(df)} 行数据")

        if 'Perovskite' not in df.columns:
            print("错误: 文件中没有找到'Perovskite'列")
            print("可用列:", df.columns.tolist())
            return None

        element_data = []
        valid_rows = []
        invalid_formulas = []

        for i, formula in enumerate(df['Perovskite']):
            if pd.isna(formula):
                invalid_formulas.append((i, "空白值"))
                continue

            formula_str = str(formula).strip()
            try:
                ratio = get_element_ratio(formula_str)
                element_data.append(ratio)
                valid_rows.append(i)
                print(f"成功解析: {formula_str}")
            except Exception as e:
                invalid_formulas.append((i, f"{formula_str} (错误: {str(e)})"))
                print(f"解析失败: {formula_str} - {str(e)}")

        ratio_df = pd.DataFrame(element_data)

        # 添加原始数据
        result_df = pd.concat([df.iloc[valid_rows].reset_index(drop=True), ratio_df], axis=1)

        print(f"\n处理摘要:")
        print(f"总行数: {len(df)}")
        print(f"成功解析: {len(valid_rows)}")
        print(f"解析失败: {len(invalid_formulas)}")

        if invalid_formulas:
            print("\n无效或无法解析的化学式:")
            for row, formula in invalid_formulas[:5]:  # 只显示前5个错误
                print(f"行 {row + 1}: {formula}")
            if len(invalid_formulas) > 5:
                print(f"...还有 {len(invalid_formulas) - 5} 个错误未显示")

        # 添加元素比例总和列
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        result_df['Bandgap'] = result_df[element_cols].sum(axis=1)

        # 保存结果
        output_dir = os.path.dirname(file_path)
        output_file = os.path.join(output_dir, "perovskite_element_ratios.xlsx")
        result_df.to_excel(output_file, index=False)
        print(f"\n结果已保存至: {output_file}")

        return result_df

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 主执行程序
if __name__ == "__main__":
    file_path = r"GeneralNullFilled.xlsx"

    print("开始解析钙钛矿化学组成...")
    print("=" * 50)

    result_df = process_perovskite_column(file_path)

    if result_df is not None:
        print("\n前5行结果预览:")
        print(result_df.head())

        # 显示基本统计信息
        print("\n元素比例统计信息:")
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        print(result_df[element_cols].describe())

        # 验证比例总和
        print("\n比例总和验证:")
        print(result_df['Bandgap'].describe())
        if (result_df['Bandgap'] < 0.95).any() or (result_df['Bandgap'] > 1.05).any():
            print("警告: 部分比例总和偏离1.0较大，请检查解析结果")

        print("\n处理完成!")