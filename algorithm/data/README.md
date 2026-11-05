# 数据集

## 来源

`cleveland.csv` 是 **UCI Cleveland 心脏病数据集**（303 例临床样本、14 个属性）的
0–4 版 `target` 文件，由项目作者提供；字段中文说明见同目录的
`data_description.txt`。

## 文件约定

- 编码 UTF-8，带表头，逗号分隔，无索引列。
- 303 行 × 14 列：`age`、`sex`、`cp`、`trestbps`、`chol`、`fbs`、`restecg`、
  `thalach`、`exang`、`oldpeak`、`slope`、`ca`、`thal`、`target`。
- `target` 取值 0–4：`0` 表示无心脏病，`1–4` 表示存在（阳性）。
  若使用原始 UCI 文件，列名 `num` 会被自动识别为 `target`。
- 缺失值写作 `?`：`ca` 有 4 条、`thal` 有 2 条。加载时统一解析为 `NaN`，
  统计时按“该列有效样本数”作分母，因此 `ca` 的占比是 `/ 299` 而不是 `/ 303`。

当前文件指纹（sha256 前 16 位）：`6fc56fce71d47cf6`。

## 替换数据

`algorithm/schema.py` 是唯一的列契约：列名、类型、取值范围、是否可缺失都在那里声明。
换数据时只要列名与取值域保持一致即可；不满足时 `algorithm.dataset.load_dataset`
会抛出 `DatasetError`，并指出缺失的列、非法数值或超范围的行号，不会静默丢数据。

替换后跑一次回归测试，黄金测试会核对页面上的核心数字：

```powershell
python -B -m unittest discover -s algorithm/tests -t .
```
