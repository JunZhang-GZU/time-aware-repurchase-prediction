# 生存分析、营销概率基准与窗口敏感性补充实验

2026-09-13在原实验环境Python 3.8.18中实际运行。未修改原始实验输出。

## 复现

1. 在仓库根目录创建Python 3.8虚拟环境，运行 pip install -r revision_20260913/requirements.txt。
2. 从 https://archive.ics.uci.edu/dataset/352/online+retail 获取数据，将Online Retail.xlsx放在仓库根目录data/raw/。
3. 在仓库根目录运行：`python revision_20260913/run_revision_experiments.py`。
4. 输出写入revision_20260913/results/，会覆盖此目录中的归档补充结果。

## 基准设计

Cox使用原17特征、原训练/验证/测试月份，以未来30天首次购买为事件，未购买者在30天右删失；标准化仅拟合训练集，固定L2惩罚0.1，不根据测试结果调参。

BG/NBD按订单去重，在每个90天窗口内构造重复购买次数x、首末单间隔tx及首单至截断点的时间T。总体参数仅使用2011-08-01训练时点拟合一次，以避免重复历史进入总体似然；固定惩罚0.01。验证、测试在各自截断点更新历史摘要，参数不更新。至少一次购买概率为P(alive)*[1-((alpha+T)/(alpha+T+h))**(r+x)]，不是预计购买次数，也不是存活概率本身。

两个新增基准均使用同一9月验证集选择F1阈值，10—11月原4612个测试样本。BG/NBD未使用完整客户获客史，窗口首单不是真实获客时间；Cox比例风险假设本次未检验。不得把点估计比较当成统计显著性结论。

## 窗口设计

60/30、90/30、180/30、90/60天四组窗口统一用6月1日训练、8月1日验证、10月1日测试，避免180天历史不足、60天标签不完整和跨集合标签重叠。

采用四组窗口共同的客户—时点样本，训练1519、验证1557、测试1762。比较RFM与完整特征逻辑回归。保留原特征计算与缺失处理，不修改原论文算法实现。此受控实验的划分和样本不同于主实验，指标不可与主实验直接相减；预测期限变化还改变标签与正类率。

结果并非所有指标均支持动态特征：60/30时PR-AUC与F1略低于RFM，180/30时F1也略低。180/30排序指标较高仅针对本次共同样本。单一测试时点不能支持普遍最优窗口或跨期稳健性。

## 检查与文件

脚本断言检查了历史/标签覆盖、跨集合标签边界、Cox事件与二分类标签对应、BG/NBD概率范围、零期限及期限单调性。checks.json记录通过状态。

additional_baselines.csv为新增基准指标；additional_predictions.csv为UCI客户标识及预测；window_sensitivity.csv为全部窗口指标（含Brier）；model_config.json为参数；cox_coefficients.csv为拟合诊断，其默认标准误未对重复客户聚类，不作为推断证据。

库文档：https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
BG/NBD实现：https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/fitters/beta_geo_fitter.py
