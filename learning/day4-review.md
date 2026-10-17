# Day 4：SFT 核心实践与 EI 复习

归档日期：2026-10-17（Asia/Shanghai）。用户指定结束本学习批次。有效学习时长未计时；此前按内容量猜测的 2–3 小时不是实测数据。

## 今天的成果与边界

- 在逐步示范和提示下写完 `05_sft_practice.py` 五个函数：compute_entropy、masked_normalize、get_response_log_probs、sft_microbatch_train_step、tokenize_prompt_and_output。尚未验收无提示独立复写。
- 助手修正 log-probs 函数多传 labels、返回键不符的两处接口问题及微批次函数一处缩进。
- 学习者亲自运行 06 单次更新和 07 累积对照并提交输出。
- 助手运行 tests/test_sft.py 的 8 项原版数值/梯度测试，8 passed、2 deselected；未改断言或快照。四个计算接口连接到练习函数。
- 分词函数用模拟分词器通过助手的小样例检查，尚未接入官方测试入口。分词与真实模型 log-probs 两项官方测试未运行；fixture 的模型路径不适用于本机。
- 助手运行 08 教学循环三次更新，loss 2.547009 → 2.230819；学习者尚未提交此脚本输出。没有真实模型基线、训练或生成正确率结果。

## 从文字到更新

文字 → 分词 → 拼接 prompt/response → padding → 移位 labels/mask
→ 模型 logits → log_softmax → gather 目标 token
→ SFT 损失 → backward → 梯度裁剪 → optimizer.step。

- logits 是原始分数，不是概率。log_softmax 稳定得到 log p，exp 还原 p。
- 熵：`-(probs * log_probs).sum(dim=-1)`，输出 [B,T]。
- gather 按 labels 编号选值，不是选最大概率；labels 已移位，不能再次移位。
- SFT 逐 token 损失是目标 log-prob 的负值；GRPO 还需优势、概率比等。
- input_ids 去掉完整序列最后一个位置；labels 和完整 mask 都去掉第一个位置。
- response_mask 对齐 labels。prompt 最后一个输入位置预测第一个回答 token，故该位置可以 mask=1。
- padding mask 为 0；排除 prompt 的预测损失不等于冻结其作为上下文参与的计算。

## 梯度累积与归一化

zero_grad 在一次有效批次前；微批次分别 backward；全部完成后裁剪，再 step。
backward 累积梯度而不更新参数；循环内清空梯度会只保留最后一份。

必须区分两种目标：

1. 教学的全局 token 平均：每份 token 损失之和除以整个有效批次 token 总数，不再除累积次数。
2. 作业微批次接口：token 损失之和 / 固定 C / 当前微批次样本数 / 累积次数。它不是默认按有效 token 数平均。

微批次平均损失只有在有效 token 数相同时，简单平均才等价于全局 token 平均；否则按 token 数加权。
尾部不足计划累积次数时需按实际数据处理；若微批次样本数也不同，还需按样本数加权。

06 验证 backward 后参数不变、step 后改变；07 中整批与累积的各参数梯度差最大约 7.5e-9。
梯度裁剪在累积完成后进行；总范数超阈值时整体缩小，与 GRPO 目标中的 clip 不同。
loss 用于 backward；detach 保留张量但断图；item 得到 Python 数字用于日志。微批次函数内部已 backward，外部不重复调用。

## 训练、验证和生成

- SFT 使用示范前缀，一次前向并行预测多个位置；正常生成逐 token 扩展自己的前缀，无 backward。
- 训练 loss 下降不直接证明生成正确率提高。
- epoch 是完整遍历训练集；optimizer step 是参数更新；累积时一个微批次不对应一次 step。
- eval 改变 dropout 等层行为，不关闭求导；no_grad 不记录计算图；验证后恢复 train。
- best 按验证指标选；latest 用于恢复。恢复训练还需优化器等状态。
- 验证集通过选择 checkpoint/超参数参与模型选择，测试集保留到方案确定后最终评估。
- 先测原模型基线；比较使用相同验证题、提示词、生成和评分设置，保存具体回答。

## EI 复习

当前权重同步给生成端 → 同题多次生成 → 评分 → 保留正确回答 → SFT → 下一轮。

- 无正确样本时记录并跳过训练，或按预先约定重采样。
- 一轮生成筛选后训练 3 epoch，仍是 1 轮 EI。
- vLLM 与训练端各有权重，更新训练端后生成端不会自动同步。
- 回答成功率和题目覆盖率分开记录；容易题可能贡献更多正确样本，全部答错的题不贡献 EI 训练样本。
- 下一主题是与 GRPO 的比较：GRPO 通常保留同题组内好坏回答，以优势控制训练方向；今天未新增 GRPO 实现或测试。

## 下次起点

1. 用五函数简图复习，先检查 08 的学习者运行情况。
2. 接入分词测试，核对本地 tokenizer 与快照是否匹配；修正模型 fixture 路径并确认实际权重，再运行剩余两项测试，不用替代模型冒充原版验收。
3. 根据验收结果推进真实模型基线、SFT；GPU 预算尚未批准，不启动付费资源。
4. EI/GRPO 继续采用完整示范后做相近练习，不突然跳到综合诊断。
