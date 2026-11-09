# algorithm —— 统计与图表数据算法

主工作台的所有数字都由这个包计算，**不含任何 Qt / PySide 依赖**，可以单独测试、
单独替换数据源。

## 结构

```
algorithm/
├── __init__.py        公开 API：load_dataset / build_workbench / ALGORITHM_VERSION
├── schema.py          列契约（类型、单位、取值范围、缺失）、年龄段、target 分桶
├── dataset.py         CSV → 类型转换 → 校验 → Dataset（含缺失统计与文件指纹）
├── stats.py           百分比、安全均值、确定性四舍五入（HALF_UP）
├── features.py        业务口径：阳性定义、性别切片、4 个风险因素谓词
├── aggregations.py    每个图表区块一个聚合函数
├── trend.py           迷你趋势：按年龄段聚合 + min-max 归一化到 0..1
├── models.py          UI 无关的视图模型（dataclass）
├── workbench.py       编排：Dataset → WorkbenchPayload
├── data/              cleveland.csv + 字段说明 + 数据约定
└── tests/             单元测试与黄金测试
```

## 用法

```python
from algorithm import load_dataset, build_workbench

dataset = load_dataset()                 # 默认读取 algorithm/data/cleveland.csv
payload = build_workbench(dataset)       # 一次算完整页

payload.total, payload.positive          # 303, 139
payload.kpis                             # 4 个 KPI（数值、单位、badge、0..1 趋势序列）
payload.age_bands.bands                  # 年龄段柱图数据 + 高亮下标
payload.positivity                       # 环图 + 性别/胸痛类型拆分
payload.target_slices                    # target 分级构成
payload.risk_factors                     # 风险因素条（含每条的分母）
payload.comparisons                      # 阳性组 / 阴性组 / 差值
```

也可以用一行拿到结果：`build_workbench_from_csv()`。

## 口径

| 项目 | 定义 |
|---|---|
| 阳性 | `target > 0`（即 1–4 级） |
| 年龄段 | `0-40`、`41-50`、`51-60`、`61-70`、`71-77`，闭区间 |
| 柱图高亮段 | 样本量 ≥ 30 的分段中阳性率最高者（避免 `71-77` 仅 6 例造成误判） |
| target 分桶 | `0`、`1`、`2-4` |
| 风险因素 | `exang == 1`、`fbs == 1`、`oldpeak > 0`、`ca > 0`，分母为该列有效样本数 |
| 迷你趋势 | 每个年龄段求均值，再做 min-max 归一到 0..1；常数序列取 0.5 |
| 取整 | 统一 HALF_UP，保留 1 位小数（与设计稿一致） |

性别、胸痛类型等切片也在 `positivity` 里给出，供后续文案生成使用。

## 测试

```powershell
python -B -m unittest discover -s algorithm/tests -t .
```

- `test_dataset`：字段类型、缺失统计、各种非法输入的报错。
- `test_aggregations`：年龄段、target 分桶、风险因素分母、组间对比。
- `test_workbench_golden`：把设计稿上的数字（303/139/45.9/54.4/131.7/139.3/158.4、
  33.3~58.9、54.1/18.2/27.7、99/45/204/123 等）逐项锁死，同时验证整条链路。

## 随机森林分析

`run_random_forest()` 是独立于 `build_workbench()` 的预测分析接口，默认使用全部特征。
每个交叉验证训练折内部先以众数填补缺失值，再对分类字段独热编码，最后拟合随机森林，
因此不会用验证折统计量填补训练数据。结果通过“样本分析 → 随机森林分析”页面查看，
与 Logistic 回归页面互不覆盖。

## 扩展

- 新增指标：在 `aggregations.py` 加函数，并把结果加进 `models.WorkbenchPayload`。
- 换配色/图标：与本包无关，只改 `ui/`。
- 缓存：数据只有 303 行，单次构建远低于 10 ms，暂不需要；如换大文件，可用
  `dataset.fingerprint` 做键缓存 payload。
