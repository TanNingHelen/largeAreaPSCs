
import numpy as np

import pandas as pd

import re
import warnings
from sklearn.preprocessing import OneHotEncoder


warnings.simplefilter(action='ignore', category=FutureWarning)

# 常量列
constantHTL = 'HTL'
constantHTL2 = 'HTL-2'
constantStructure = 'Structure'
constantHTL_Passivator = 'HTL_Passivator'
constantHTL_Addictive = 'HTL-Addictive'
constantETL = 'ETL'
constantETL2 = 'ETL-2'
constantETL_Passivator = 'ETL_Passivator'
constantETL_Addictive = 'ETL-Addictive'
constantMetalElectrode = 'Metal_Electrode'
constantGlass = 'Glass'
constantPerovskite = 'Perovskite'
constantPrecursorSolutionAddictive = 'Precursor_Solution_Addictive'
constantPrecursorSolution = 'Precursor_Solution'
constantAntisolvent = 'Antisolvent'
constantAnnealingTemperature1 = 'Annealing_Temperature1'
constantAnnealingTemperature2 = 'Annealing_Temperature2'
constantAnnealingTimeMin = 'Annealing_Time1'
constantAnnealingTime2 = 'Annealing_Time2'
constantDepositionMethod = 'Deposition_Method'

constantP1WavelengthNm = 'P1Wavelength(nm)'
constantP2WavelengthNm = 'P2Wavelength(nm)'
constantP3WavelengthNm = 'P3Wavelength(nm)'
constantTotalScribingLineWidth = 'total_scribing_line_width(μm)'
constantP1Wide = 'P1Width(μm)'
constantP2Wide = 'P2Width(μm)'
constantP3Wide = 'P3Width(μm)'
constantType = 'Type'
constantGFF = 'GFF'
constantsub='submodule_number'

constantP1v = 'P1Scan_Velocity(mm/s)'
constantP1frequency = 'P1etching_frequency(kHz)'
constantP1SpotSize = 'P1Spot Size(μm)'
constantP1Power = 'P1etching_Power(W)'
constantP1Powerpercent = 'P1etching_Power_percentage(%)'
constantP2v = 'P2Scan_Velocity'
constantP2frequency = 'P2etching_frequency(kHz)'
constantP2SpotSize = 'P2Spot Size(μm)'
constantP2Power = 'P2etching_Power(W)'
constantP2Powerpercent = 'P2etching_Power_percentage(%)'
constantP3v = 'P3Scan_Velocity'
constantP3frequency = 'P3etching_frequency(kHz)'
constantP3SpotSize = 'P3Spot Size(μm)'
constantP3Power = 'P3etching_Power(W)'
constantP3Powerpercent = 'P3etching_Power_percentage(%)'
constantP1P2='P1_P2Scribing_Spacing(μm)'
constantP2P3 = 'P2_P3Scribing_Spacing(μm)'
constantbrand = 'brand'




# -----数据操作-----

# step1：
# 导入目标数据集
df = pd.read_excel("OriginData.xlsx")

#先处理行

# 定义需要检查的列名列表
cols_to_check = [
    constantP1WavelengthNm,  # 使用变量而不是字符串
    constantP2WavelengthNm,
    constantP3WavelengthNm,
    constantTotalScribingLineWidth,
    constantP1Wide,
    constantP2Wide,
    constantP3Wide
]

# 生成布尔掩码：当所有指定列都为 null 时返回 True
mask_all_null = df[cols_to_check].isna().all(axis=1)
df = df[~mask_all_null]

#再处理列
# HTL列，空值或者null值，用0处理
df[constantHTL] = df[constantHTL].fillna('0')

# HTL-2列，空值或者null值，用0处理
df[constantHTL2] = df[constantHTL2].fillna('0')

# HTL-Passivator列，空值或者null值，用0处理
df[constantHTL_Passivator] = df[constantHTL_Passivator].fillna('0')

# HTL-Addictive列，空值或者null值，用0处理
df[constantHTL_Addictive] = df[constantHTL_Addictive].fillna('0')

# ETL列，空值或者null值，用0处理
df[constantETL] = df[constantETL].fillna('0')

# ETL-2列，空值或者null值，用0处理
df[constantETL2] = df[constantETL2].fillna('0')

# ETL-Passivator列，空值或者null值，用0处理
df[constantETL_Passivator] = df[constantETL_Passivator].fillna('0')

# ETL_Addictive列，空值或者null值，用0处理
df[constantETL_Addictive] = df[constantETL_Addictive].fillna('0')

# MetalElectrode列，空值或者null值，用0处理
df[constantMetalElectrode] = df[constantMetalElectrode].fillna('0')

# Perovskite列，空值或者null值，用0处理
df[constantPerovskite] = df[constantPerovskite].fillna('0')

# PrecursorSolution列，空值或者null值，用0处理
df[constantPrecursorSolution] = df[constantPrecursorSolution].fillna('0')

# PrecursorSolutionAddictive列，空值或者null值，用0处理
df[constantPrecursorSolutionAddictive] = df[constantPrecursorSolutionAddictive].fillna('0')

# Antisolvent列，空值或者null值，用0处理
df[constantAntisolvent] = df[constantAntisolvent].fillna('0')

# 获取众数
mode_AnnealingTimeMin = df[constantAnnealingTimeMin].mode()
mode_ConstantAnnealingTemperature1 = df[constantAnnealingTemperature1].mode()

# 用众数填充
df[constantAnnealingTimeMin] = df[constantAnnealingTimeMin].fillna(mode_AnnealingTimeMin.iloc[0])
df[constantAnnealingTemperature1] = df[constantAnnealingTemperature1].fillna(mode_ConstantAnnealingTemperature1.iloc[0])

# AnnealingTemperature2列，空值或者null值，用0处理
df[constantAnnealingTemperature2] = df[constantAnnealingTemperature2].fillna('0')

# AnnealingTime2列，空值或者null值，用0处理
df[constantAnnealingTime2] = df[constantAnnealingTime2].fillna('0')



# Deposition Method处理， 使用众数进行处理
mode_constantDepositionMethod = df[constantDepositionMethod].mode()
print('Deposition Method 众数: ')
print(mode_constantDepositionMethod.iloc[0])
df[constantDepositionMethod] = df[constantDepositionMethod].fillna(mode_constantDepositionMethod.iloc[0])

# P1WavelengthNm列，空值或者null值，用0处理
df[constantP1WavelengthNm] = df[constantP1WavelengthNm].fillna('0')

# P2WavelengthNm列，空值或者null值，用0处理
df[constantP2WavelengthNm] = df[constantP2WavelengthNm].fillna('0')

# P3WavelengthNm列，空值或者null值，用0处理
df[constantP3WavelengthNm] = df[constantP3WavelengthNm].fillna('0')

# total scribing line width列，空值或者null值，用0处理
df[constantTotalScribingLineWidth] = df[constantTotalScribingLineWidth].fillna('0')



# P1Wide列，空值或者null值，用0处理
df[constantP1Wide] = df[constantP1Wide].fillna('0')

# P2Wide列，空值或者null值，用0处理
df[constantP2Wide] = df[constantP2Wide].fillna('0')

# P3Wide列，空值或者null值，用0处理
df[constantP3Wide] = df[constantP3Wide].fillna('0')

# Type列，空值或者null值，用0处理
df[constantType] = df[constantType].fillna('0')

# SubmoduleNumber列，空值或者null值，用0处理
df[constantsub] = df[constantsub].fillna('0')

# GFF列，空值或者null值，用0处理
df[constantGFF] = df[constantGFF].fillna('0')


#P1刻蚀速度列，空值或者null值，用0处理
df[constantP1v] = df[constantP1v].fillna('0')
#P1刻蚀功率列空值或者null值，用0处理
df[constantP1frequency] = df[constantP1frequency].fillna('0')
#P1光斑大小
df[constantP1SpotSize] = df[constantP1SpotSize].fillna('0')
#P1功率
df[constantP1Power] = df[constantP1Power].fillna('0')
#P1功率比
df[constantP1Powerpercent] = df[constantP1Powerpercent].fillna('0')

#P2刻蚀速度列，空值或者null值，用0处理
df[constantP2v] = df[constantP2v].fillna('0')
#P2刻蚀功率列空值或者null值，用0处理
df[constantP2frequency] = df[constantP2frequency].fillna('0')
#P2光斑大小
df[constantP2SpotSize] = df[constantP2SpotSize].fillna('0')
#P2功率
df[constantP2Power] = df[constantP2Power].fillna('0')
#P2功率比
df[constantP2Powerpercent] = df[constantP2Powerpercent].fillna('0')

#P3刻蚀速度列，空值或者null值，用0处理
df[constantP3v] = df[constantP3v].fillna('0')
#P3刻蚀功率列空值或者null值，用0处理
df[constantP3frequency] = df[constantP3frequency].fillna('0')
#P3光斑大小
df[constantP3SpotSize] = df[constantP3SpotSize].fillna('0')
#P3功率
df[constantP3Power] = df[constantP3Power].fillna('0')
#P3功率比
df[constantP3Powerpercent] = df[constantP3Powerpercent].fillna('0')
#P1P2距离
df[constantP1P2] = df[constantP1P2].fillna('0')
#P2P3距离
df[constantP2P3] = df[constantP2P3].fillna('0')
#激光品牌
df[constantbrand] = df[constantbrand].fillna('0')




null_columns = [col for col in df.columns if df[col].isnull().any() or df[col].eq('').any()]
print("包含null值或空值的列名：")
print(null_columns)

# 初始化编码器
encoder = OneHotEncoder(
    sparse_output=False,  # 返回密集数组
    handle_unknown='ignore',  # 处理未知类别
    drop='first'  # 避免虚拟变量陷阱（删除第一个类别）
)

# 执行编码
encoded_array = encoder.fit_transform(df[[constantStructure]])



# 将处理后的数据保存为xlsx文件
df.to_excel('GeneralNullFilled1.xlsx', index=False)
