# Predicting Airline Satisfaction — playground-series-s6e10

Kaggle Playground S6E10：二分类，预测乘客是否满意（`satisfaction`），提交概率，按 ROC AUC 评估。
train 699,635 行 / test 299,844 行 / 22 列原始特征。

**训练全部在 Kaggle Notebook 上执行**；本地只负责生成 notebook、跑 smoke 验证、取回结果。

## 目录

```
src/airline_satisfaction.py         notebook 源文件（percent 格式，同时也是可直接执行的 .py）
src/make_notebook.py                生成 kaggle/airline_satisfaction.ipynb（顺带语法检查每个 code cell）
src/smoke_check.py                  本地 5 万行抽样跑通全流程，不训练全量
src/exp_domain.py                   跨域诊断：原始数据 vs 合成数据（需先下载原始数据集，见下）
kaggle/kernel-metadata.json         kernel 配置（私有 CPU；比赛数据 + 原始数据集两个来源）
kaggle/airline_satisfaction.ipynb   生成物，push 的实际内容
data/                               本地原始数据（gitignore）
outputs/                            Kaggle 产出与日志（gitignore）
```

## 在 Kaggle 上跑

```powershell
.\.venv\Scripts\python.exe src\make_notebook.py          # 改了 .py 必须重新生成 ipynb
.\.venv\Scripts\kaggle.exe kernels push -p kaggle
.\.venv\Scripts\kaggle.exe kernels status jianansun/predicting-airline-satisfaction
.\.venv\Scripts\kaggle.exe kernels output jianansun/predicting-airline-satisfaction -p outputs
```

产出落在 `outputs/`：`submission.csv`、`metrics.json`、运行日志。提交成绩：

```powershell
.\.venv\Scripts\kaggle.exe competitions submit -c playground-series-s6e10 -f outputs\submission.csv -m "lgb+xgb blend"
```

## 模型结构

1. **特征工程**（22 → 47 列）
   - 13 个服务评分的均值 / 标准差 / 极值 / 0、4、5 的计数 / 0 的占比
   - 分组体验：数字渠道（WiFi、在线订票、在线登机）与舒适度（座椅、腿部空间、清洁、餐食）
   - 延误结构：到达延误缺失标记、总延误、延误差、是否延误、对数
   - 交互与分箱：`Class × Type of Travel`、`Customer Type × Class`、`Gender × Type of Travel`、年龄 / 距离 / 延误分箱
2. **三个模型配置，同一套 10 折 StratifiedKFold**
   - `lgb_a`：`num_leaves=127`、`lr=0.03`、`ff=0.65`，每折内 2 个 seed 平均
   - `lgb_b`：`num_leaves=255`、`lr=0.02`、`ff=0.45`、`min_child_samples=250`，制造分歧
   - `xgb`：`depth=8`、`lr=0.04`，提供跨库多样性
   - CatBoost 有实现但默认关闭（`USE_CATBOOST`），原因见实验记录
3. **补充训练数据**：可选叠加原始 Invistico 数据集（`teejmahal20/airline-passenger-satisfaction`）。只进入训练集、不参与验证，因此 OOF AUC 仍可比；实测无增益，默认关闭（`USE_ORIGINAL_DATA`）
4. **融合**：各模型预测先转 rank，再用 Caruana 贪心选择（每轮加一个让 OOF AUC 涨最多的模型，最多 30 轮，允许重复）；单模型最优时退化为单模型
5. **提交**：`id, satisfaction`，概率在 [0,1]

## 运行记录

| 版本 | 模型 | 折数 / seed | 特征数 | 补充数据 | OOF AUC | 公开 LB |
|---|---|---|---|---|---|---|
| v1 | LGB + XGB | 5 / 1 | 41 | 否 | 0.958957 | 0.95846 |
| v2 | LGB + XGB + CatBoost | 5 / 1 | 47 | 是 | 0.958846 | — |
| v3 | LGB + XGB（v1 参数） | 5 / 1 | 47 | 否 | 0.958884 | 0.95834 |
| v4 | `lgb_a` + `lgb_b` + `xgb`，贪心集成 | 10 / 2 | 47 | 否 | 0.959238 | 0.95845 |
| v5 | `lgb_a` + `lgb_b` + `nn`（GPU） | 10 / 1-2 | 47 | 否 | 0.959289 | 0.95843 |
| **FE** | LGBM + digit/freq/TE/原始预测 | 5 / 1 | 214 | 原始 | 0.960296 | 0.95986 |
| **Stack** | 公开 OOF 成员的 LR 堆叠（30 → 72 → 77 成员） | 5 / — | — | — | **0.961961** | **0.96167** |
| FE v3 | FE + 航线画像 + 航线/评分×分段 TE | 5 / 1 | 296 | 原始 | 0.960828 | 0.96077 |
| Stack v5 | 252 成员，family L2（C=0.003）+ 0.2 LightGBM | 5 / — | — | — | 0.962235 | 0.96184 |
| Stack v6 | 同 v5 成员池，自建成员换成 FE v3；融合未过 4/5 折门槛，退回纯 LR C=0.003 | 5 / — | — | — | 0.962216 | **0.96185** |
| RealMLP | pytabkit RealMLP：`base`（复现公开配置）/ `route`（+ 航线特征） | 5 / 8 ens | 49 / 79 | 否 | 0.961217 / 0.961179 | 0.96106 / 0.96106 |
| Stack v7 | v6 + 两个 RealMLP 成员（254 成员），0.8 LR + 0.2 LGB（flat C=0.003） | 5 / — | — | — | 0.962224 | **0.96185** |
| FE v4 | FE v3 + 完整航线画像 + 辅助任务期望值 + 更强正则（127 叶 / min_child 50 / ff 0.5） | 5 / 1 | 331 | 原始 | 0.961000 | 0.96100 |
| RealMLP large | 网络加宽一倍（1024-512-256）、6 epoch | 5 / 8 ens | 49 | 否 | 0.961034 | 0.96101 |
| RealMLP seeds3 | 公开配置 × 3 seed 平均 | 5 / 24 ens | 49 | 否 | 0.961265 | 0.96107 |
| Stack v8 | v7 成员池，自建成员换成 FE v4 / RealMLP large / seeds3；融合未过门槛，纯 LR C=0.003 | 5 / — | — | — | 0.962217 | **0.96186** |
| Transformer | 行即 token 小型 Transformer（24 token、0.58M 参数、3 epoch × 2 seed） | 5 / 2 | — | 原始 | 0.960570 | 0.96035 |
| Stack v9 | v8 + Transformer 成员（255 成员），纯 LR C=0.003 | 5 / — | — | — | 0.962215 | 0.96186 |

v2 相对 v1 掉 0.0001（噪声量级），但 CatBoost 5 折单吃掉 2.1 小时、自身 OOF 只有 0.958439。
因此 v3 把 CatBoost 与外部数据都做成默认关闭的开关；v3 与 v1 的 0.00007 差同样在噪声内，说明那 6 个新增特征没有实质贡献。
v4 换 10 折 + 折内 seed 平均 + 贪心集成，CV 提升 0.00028（0.95896 → 0.95924），**但 LB 完全没动**（0.95846 → 0.95845）。
v5 加入带类别嵌入的 MLP，CV 到 0.959289，LB 仍是 0.95843——GBDT/MLP 这一族做法已经饱和。

**FE 版是分水岭**：换成公开 notebook 验证过的特征组合（digit、频次/rarity、折内目标编码、原始数据先验），
OOF 直接到 0.960296、LB 到 0.95986，`+0.0014`。所以之前的「天花板」判断是错的——瓶颈在特征，不在模型族。

**Stack 版是第二条曲线**：把竞赛里公开的 OOF 预测库（同一套 5 折）当特征做 LR 堆叠。
30 成员 OOF 0.961825 / LB 0.96145，72 成员 OOF 0.961957 / LB 0.96167，77 成员 OOF 0.961961（已饱和）。

**FE v3**：精确 `Flight Distance` 当航线 ID，加航线画像（同距离下评分均值与个人偏差）和航线 / 评分 × 乘客分段的折内 TE，
OOF 0.960296 → 0.960828（+0.00053）；`TE_route_class`、`TE_route_travel`、`route_*_delta` 进入 gain 前十。

**Stack v5**：成员池扩到 252（去掉 84 个近似重复与 stack 级成员），C 从 1 降到 0.003，family-wise L2，再融 0.2 的 LightGBM 二级模型，
嵌套 CV 0.961961 → 0.962235，5/5 折胜过纯 LR 基线，LB 0.96167 → 0.96184。自建 FE 成员（v2 输出）只贡献 +2.9e-6。

FE v3 单模型 LB 0.96077（FE v2 0.95986），OOF 与 LB 只差 0.00006，航线特征在测试集上同样成立。
但把它换进堆叠（v6）只贡献 +2.3e-6：252 个成员里已有大量 route-ID 类模型，GBDT 再强也几乎没有新增信息。
v6 的 LightGBM 融合只在 2/5 折胜出，按规则退回纯 LR；v5 / v6 的 LB 差 0.00001，属于公开榜噪声（±1.5e-5）。

**RealMLP（神经网络成员）**：`base` 精确复现公开 notebook（OOF 0.961217，T4 上 9 分钟）。
`route` 加入航线画像与航线 TE 后反而略降（0.961179，1/5 折更好）：RealMLP 已把精确 `Flight Distance` 当类别做 embedding，
等于直接学到航线 ID，航线画像是重复信息；GBDT 不擅长 3500+ 基数的类别，所以同样的特征对 FE LightGBM 有效。
放进堆叠（v7）后逐个去掉的增量：FE v3 +2.1e-6、RealMLP base 0.0、route −2.9e-6；v7 嵌套 CV 0.962224 与 v5 持平，LB 0.96185 与 v6 相同；两个 RealMLP 单模型 LB 都是 0.96106。
结论：池里已有 161 GBDT / 60 MLP / 17 TabPFN / 14 TFM，单个 0.961 级的自建模型对堆叠的边际贡献 ≈ 0，堆叠在 ~0.96223 饱和。

**更大的模型无效**：RealMLP 加宽一倍 + 6 epoch，OOF 0.961217 → 0.961034（5 折全降，LB 0.96101）；
3 seed 平均只有方差收益 +0.00005（LB 0.96107 vs 0.96106）。标签噪声决定上限，容量不是瓶颈。

**FE v4：特征仍然有效**（参考 `goodpjw2008/s6e10-auxiliary-task-features`）：
- 完整航线画像：同一距离下全部 20 个原始列的均值 + 行数（v3 只有 6 列）
- 辅助任务特征：train+test 合并，用其余 20 列预测 13 个评分与 3 个类别列（多分类 XGBoost，3 折，GPU），
  取评分期望值 `aux_ev_*` 与整行伪对数似然 `aux_sum_logp`；不用标签。7 个 `aux_ev_*` 进 gain 前 15
- LightGBM 更强正则：127 叶、`min_child_samples=50`、`feature_fraction=0.5`

OOF 0.960828 → 0.961000（5/5 折上升），LB 0.96077 → 0.96100（+0.00023），是目前最好的自建 GBDT。

放进堆叠（v8）：FE v4 +2.6e-6、RealMLP large −0.3e-6、seeds3 +0.4e-6，合计 +2.7e-6；嵌套 CV 0.962217 与 v6 相同，
LB 0.96186（与 v6 / v7 的 0.96185 差 1e-5，属噪声）。单模型提升 0.0002 也换不来堆叠增量。

**小型 Transformer（`src/airline_transformer.py`）**：参考 Chris Deotte 讨论区的 Q-FT / ModernBERT 方案，
每列一个 token（列嵌入 + 取值查表 + 数值 sin/cos 周期嵌入 + 4 个侧特征），另加行统计与原始数据先验两个摘要 token，
无位置编码（把一行读成集合）。6 epoch 会过拟合（fold 1 验证 AUC 第 3 epoch 见顶 0.96074 → 第 6 epoch 0.95993），
3 epoch × 2 seed 后 OOF 0.960570 / LB 0.96035，T4 上训练仅 8 分钟。
放进堆叠（v9）增量 −1.3e-6（1/5 折）：模型族虽新，信息已被池内成员覆盖。
自建成员在堆叠里的排序：FE v4 +4.6e-6 > RealMLP large +0.4e-6 > seeds3 −0.9e-6 > Transformer −1.3e-6，全部在噪声内。
v9 嵌套 CV 0.962215 未超过 v5 / v8；LB 0.96186 与 v8 相同，最终选择维持 Stack v5 + v8。
注意它的分数主要来自社区模型，不是自建能力：成员平均 0.961373、最强单成员 0.961388，
所以堆叠本身带来的增量约 +0.0006，其余是成员本身的水平。
另一个风险是 private LB：这些成员共享同一份公开测试集信号，公开分领先不代表私榜同样领先。

## 诊断：这题的天花板在哪

`src/exp_domain.py` 用同一套小特征 + 短 LightGBM 做跨域对比：

| 训练 → 验证 | AUC |
|---|---|
| 原始 Invistico → 合成验证集 | 0.9541 |
| 合成 → 原始验证集 | **0.9936** |
| 合成 → 合成验证集（即 CV 参照） | 0.9579 |
| 原始 + 合成 → 合成验证集 | 0.9580 |

结论：

1. **合成数据由原始数据生成，规则高度可学**——合成训练的模型几乎完美预测原始标签（0.9936）。
2. **合成标签含噪声**，所以合成内部 AUC 上限就在 0.958–0.96 附近。榜首公开 LB 0.96309 与这个上限一致，说明剩下 0.004 的空间属于"把 P(y|x) 估得更准"，不是找隐藏特征。
3. **叠加原始数据没有增益**（+0.00003）：它带来的信息已经全在合成数据里了，反而稀释样本权重。
4. 已排除常见泄漏：训练集内无重复特征向量，测试集与训练集特征向量零重叠，`id` 单独 AUC = 0.5001。

复现诊断需要原始数据集（已 gitignore，需自己下载）：

```powershell
.\.venv\Scripts\kaggle.exe datasets download -d teejmahal20/airline-passenger-satisfaction -p data\orig --unzip
.\.venv\Scripts\python.exe src\exp_domain.py
```

注意该数据集的文件也叫 `train.csv` / `test.csv`，务必解压到 `data\orig\`，否则会覆盖比赛数据。

## 参考公开方案（用 `kaggle kernels pull` 拉下来读的）

| notebook | 分数 | 关键信息 |
|---|---|---|
| [evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96111](https://www.kaggle.com/code/evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96111) | LB 0.96111 | 逐项 ablation：digit +0.00069、频次 +0.00038、TE +0.00038，是真实增量来源 |
| [yekenot/ps-s6-e10-realmlp-pytabkit](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit) | — | `RealMLP_TD_Classifier`（pytabkit）+ 数值全转类别 + count 编码 + 组合键 TE |
| [cdeotte/stacking-tfms-with-public-oof](https://www.kaggle.com/code/cdeotte/stacking-tfms-with-public-oof) | — | 公开 OOF 库（124 个模型的 OOF/test 对，同一套 5 折）做二级 stacking |
| [rabari9999/s6e10-lb-0-96207-rank-16](https://www.kaggle.com/code/rabari9999/s6e10-lb-0-96207-rank-16) | LB 0.96207 | **不是模型**：对公开 LB 拟合二次响应面做梯度爬坡；作者自述 private 会掉，不建议学 |

结论：榜上 0.962 里有相当一部分是对公开 LB 的过拟合；能复现的真实提升来自特征（digit / 频次 / TE / 原始数据先验）。

## 工作约定

每次向 LB 提交 `submission.csv` 之前，先把对应代码 commit 掉，保证榜上的分数总能追溯到具体 commit。

## 踩过的坑

- **比赛数据挂载路径**：新环境是 `/kaggle/input/competitions/<slug>/`，老教程里写的 `/kaggle/input/<slug>/` 已经不对了。notebook 里 `resolve_competition_dir()` 两种都兼容，并会递归兜底。
- **中文日志乱码**：Kaggle 日志按本地码页截断，运行期 `print` 全部用英文，中文只留在 markdown cell 里。
- **jupytext 在本机不可用**：`jupytext`/`nbformat` 会往用户目录写签名状态，被项目沙箱拦住，所以改用 `src/make_notebook.py` 自己解析 percent 格式。
- **`uv` 缓存目录**：默认缓存在用户目录，沙箱下不可写；本地装包前需要 `$env:UV_CACHE_DIR = "$PWD\.uv-cache"`。

## 待办

- 10 折 + 多 seed 平均：标签有噪声时这是最稳的正收益，能压低 OOF 方差、让融合权重更可信
- OOF 二级 stacking（LogisticRegression / 浅层 LGBM）
- 直接优化 AUC 的替代损失（XGBoost `rank:pairwise`），噪声标签下常比 logloss 更贴指标
- CatBoost 若要重启：把 `iterations` 降到 1200、`depth` 降到 7，单折可压到 10 分钟内
- 确认赛题是否允许外部数据；即使允许，本例已验证无增益，保持 `USE_ORIGINAL_DATA = False`
