# h1
## h2
### h3 h4

# h1 ML_for_Perovskite_Laser_Scribing/

```text
ML_for_Perovskite_Laser_Scribing/
├── data/                           # 数据目录
│   ├── raw/                        # 原始收集数据
│   ├── processed/                  # 清洗和特征工程后的数据
│   └── dataset_description.md      # 数据集详细说明
├── src/                            # 源代码目录
│   ├── 01_data_preprocessing.py    # 数据清洗、编码、缺失值处理
│   ├── 02_feature_engineering.py   # 特征构建与计算
│   ├── 03_model_training.py        # 模型训练与超参数优化
│   ├── 04_model_evaluation.py      # 模型评估与可视化
│   ├── 05_shap_analysis.py         # SHAP可解释性分析
│   ├── 06_parameter_optimization.py # 参数优化（子模块宽度、激光参数）
│   └── utils.py                    # 通用工具函数
├── notebooks/                      # Jupyter Notebook分析
│   ├── 01_EDA.ipynb                # 探索性数据分析
│   └── 02_Laser_Parameter_Analysis.ipynb # 激光参数深入分析
├── configs/                        # 配置文件
│   ├── hyperparameters.yaml        # 模型超参数配置
│   └── paths.yaml                  # 文件路径配置
├── results/                        # 输出结果
│   ├── figures/                    # 生成的所有图表
│   ├── tables/                     # 性能表格
│   └── models/                     # 保存的模型文件
├── requirements.txt                # Python依赖包列表
├── run_pipeline.py                 # 主运行脚本（一键复现）
└── README.md                       # 本文件
```