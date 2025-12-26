import pandas as pd
import re
import os
from collections import defaultdict


def preprocess_formula(formula):
    """预处理化学式，处理常见格式问题"""
    if pd.isna(formula):
        return formula

    formula = str(formula).strip()

    # 移除不可见字符和特殊空白字符
    formula = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', formula)
    formula = re.sub(r'\s+', '', formula)  # 移除所有空格

    # 处理换行符
    formula = formula.replace('\n', '').replace('\r', '')

    # 处理上标数字
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for sub, normal in subscript_map.items():
        formula = formula.replace(sub, normal)

    # 处理数学点
    formula = formula.replace('⋅', '.')

    # 修复：更安全地处理小写l，避免影响Cl
    # 只在l前面不是C且后面是数字时替换
    formula = re.sub(r'(?<![Cc])l(?=\d)', 'I', formula)

    # 处理方括号（转换为圆括号）
    formula = formula.replace('[', '(').replace(']', ')')

    # 处理有机阳离子的完整名称
    formula = re.sub(r'\[CH\(NH2\)2\]', 'FA', formula)
    formula = re.sub(r'CH\(NH2\)2', 'FA', formula)
    formula = re.sub(r'\[CH3NH3\]', 'MA', formula)
    formula = re.sub(r'CH3NH3', 'MA', formula)

    return formula


def parse_simple_formula(formula):
    """解析简单化学式（无括号）"""
    elements = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
    ratios = {ele: 0.0 for ele in elements}

    # 按元素长度降序排列，确保先匹配长元素名（如MA, FA）
    sorted_elements = sorted(elements, key=len, reverse=True)

    remaining_formula = formula

    for ele in sorted_elements:
        # 匹配元素及其后面的数字
        pattern = re.escape(ele) + r'(\d*\.?\d*)'
        match = re.search(pattern, remaining_formula)

        if match:
            num_str = match.group(1)
            if not num_str:
                ratios[ele] = 1.0
            else:
                ratios[ele] = float(num_str)

            # 从剩余字符串中移除已匹配的部分
            start, end = match.span()
            remaining_formula = remaining_formula[:start] + remaining_formula[end:]

    return ratios, remaining_formula


def get_element_ratio(composition):
    elements = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
    element_ratio = {ele: 0.0 for ele in elements}

    # 预处理化学式
    composition = preprocess_formula(composition)

    # 处理括号内容
    if '(' in composition:
        # 匹配括号内容和乘数
        pattern = r"\(([^()]+)\)(\d*\.?\d*)"
        matches = re.findall(pattern, composition)

        for bracket_content, multiplier_str in matches:
            if not multiplier_str:
                multiplier = 1.0
            else:
                multiplier = float(multiplier_str)

            # 解析括号内的元素
            bracket_ratios, _ = parse_simple_formula(bracket_content)

            # 将括号内的比例乘以乘数并累加
            for ele in elements:
                element_ratio[ele] += bracket_ratios[ele] * multiplier

            # 从原字符串中移除已处理的括号部分
            bracket_pattern = re.escape(f"({bracket_content}){multiplier_str}")
            composition = re.sub(bracket_pattern, '', composition)

    # 解析剩余部分（无括号）
    remaining_ratios, _ = parse_simple_formula(composition)
    for ele in elements:
        element_ratio[ele] += remaining_ratios[ele]

    # 处理特殊情况：如果Pb存在但没有明确比例，设为1
    if any(element_ratio[ele] > 0 for ele in ['Cs', 'MA', 'FA', 'Rb']) and \
            any(element_ratio[ele] > 0 for ele in ['I', 'Br', 'Cl']) and \
            element_ratio['Pb'] == 0 and element_ratio['Sn'] == 0:
        element_ratio['Pb'] = 1.0

    return element_ratio


# 处理Excel文件
def process_perovskite_column(file_path):
    try:
        # 检查输入文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 输入文件 {file_path} 不存在!")
            print(f"当前工作目录: {os.getcwd()}")
            return None

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
            except Exception as e:
                invalid_formulas.append((i, f"{formula_str} (错误: {str(e)})"))

        # 检查是否有有效数据
        if len(element_data) == 0:
            print("错误: 没有成功解析任何化学式!")
            return None

        ratio_df = pd.DataFrame(element_data)

        # 确保所有必要的列都存在
        required_columns = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        for col in required_columns:
            if col not in ratio_df.columns:
                ratio_df[col] = 0.0

        # 添加原始数据
        result_df = pd.concat([df.iloc[valid_rows].reset_index(drop=True), ratio_df], axis=1)

        print(f"\n处理摘要:")
        print(f"总行数: {len(df)}")
        print(f"成功解析: {len(valid_rows)}")
        print(f"解析失败: {len(invalid_formulas)}")
        print(f"成功率: {len(valid_rows) / len(df) * 100:.1f}%")

        # 特别统计Cl的情况
        cl_count = (ratio_df['Cl'] > 0).sum()
        print(f"包含Cl元素的样本数: {cl_count}")
        if cl_count > 0:
            print(f"Cl比例范围: [{ratio_df['Cl'].min():.4f}, {ratio_df['Cl'].max():.4f}]")

        if invalid_formulas:
            print("\n无效或无法解析的化学式:")
            for row, formula in invalid_formulas[:10]:  # 只显示前10个错误
                print(f"行 {row + 1}: {formula}")
            if len(invalid_formulas) > 10:
                print(f"...还有 {len(invalid_formulas) - 10} 个错误未显示")

        # 计算比例总和（不强制归一化）
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        result_df['Total_Ratio'] = result_df[element_cols].sum(axis=1)

        print("\n注意: 'Total_Ratio' 列是元素比例原始总和，未进行归一化")

        # 保存结果
        output_file = "perovskite_element_ratios2.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"\n结果已保存至: {output_file}")

        # 验证文件是否创建成功
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"文件创建成功! 文件大小: {file_size} 字节")
            print(f"文件位置: {os.path.abspath(output_file)}")
        else:
            print("错误: 文件保存后未找到!")
            return None

        return result_df

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 主执行程序
if __name__ == "__main__":
    # 待解析文件路径
    file_path = r"GeneralNullFilled2.xlsx"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        print(f"当前工作目录: {os.getcwd()}")
        print("请确保文件存在于当前目录")
        exit(1)

    print("开始解析钙钛矿化学组成...")
    print("=" * 50)

    result_df = process_perovskite_column(file_path)

    if result_df is not None:
        print("\n前5行结果预览:")
        # 只显示元素比例相关的列
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl', 'Total_Ratio']
        display_cols = [col for col in element_cols if col in result_df.columns]
        if display_cols:
            print(result_df[display_cols].head().round(4))
        else:
            print(result_df.head())

        # 显示基本统计信息
        print("\n元素比例统计信息:")
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        available_cols = [col for col in element_cols if col in result_df.columns]
        if available_cols:
            stats = result_df[available_cols].describe().round(4)
            print(stats)

        # 验证比例总和
        print("\n比例总和验证:")
        print(result_df['Total_Ratio'].describe().round(4))

        print("\n处理完成!")
    else:
        print("\n处理失败，没有生成结果!")