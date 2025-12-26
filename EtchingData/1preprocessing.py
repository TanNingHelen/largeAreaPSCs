import pandas as pd
from sklearn import preprocessing
import os

# 创建目录保存映射关系
os.makedirs("label_mappings", exist_ok=True)

# 读取原始数据
df = pd.read_excel(r"BandgapDone1.xlsx")
print(f"原始数据行数: {len(df)}")

# ********** 修改开始 **********
# 备份原始数据，不进行 groupby 操作
df1 = df.copy()  # 直接使用原始数据副本
print(f"处理后数据行数: {len(df1)}")
# ********** 修改结束 **********

# 创建存储映射关系的字典
label_mappings = {}

# 字符串映射成数字
categorical_cols = ['Structure', 'HTL', 'HTL-2', 'HTL_Passivator','HTL-Addictive', 'ETL', 'ETL-2',
                    'ETL_Passivator','ETL-Addictive', 'Metal_Electrode', 'Glass', 'Precursor_Solution'
                ,  'Precursor_Solution_Addictive','Deposition_Method', 'Antisolvent', 'Type','brand']

for col in categorical_cols:
    lbl = preprocessing.LabelEncoder()
    df1[col] = lbl.fit_transform(df1[col].astype('str'))

    # 保存映射关系
    mapping = pd.DataFrame({
        'Original': lbl.classes_,
        'Encoded': range(len(lbl.classes_))
    })
    label_mappings[col] = mapping

    # 保存到CSV
    mapping.to_csv(f"label_mappings/{col}_mapping.csv", index=False)
    print(f"已保存 {col} 的映射关系")

# 保存所有映射关系的汇总表
mapping_summary = []
for col, mapping_df in label_mappings.items():
    temp_df = mapping_df.copy()
    temp_df['Feature'] = col
    mapping_summary.append(temp_df)

full_mapping = pd.concat(mapping_summary)
full_mapping.to_csv("label_mappings/full_mapping_summary.csv", index=False)

# 去除没有相关性的特征
df1 = df1.drop(columns=['Sn','Rb'])

# 输出清理后的数据到Excel
df1.to_excel(r"FinalData1.xlsx", index=False)

print("\n数据清理完成，已保存为 FinalData1.xlsx")
print(f"最终数据行数: {len(df1)}")
print("标签映射关系已保存至 label_mappings 目录：")
print("- 各特征的独立映射文件: label_mappings/[feature]_mapping.csv")
print("- 完整映射汇总表: label_mappings/full_mapping_summary.csv")