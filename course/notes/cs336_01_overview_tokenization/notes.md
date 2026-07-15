# CS336 Lecture 1：课程导论与分词

> **讲师**：Percy Liang, Tatsu Hashimoto（Stanford）
> **时长**：~79 分钟 ｜ **日期**：Spring 2025
> **范围**：课程动机 → 五大模块总览 → 分词（tokenization）深入
> **视频**：[YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)（第 1 集）
> **字幕来源**：YouTube auto-sub（已去重清洗）

---

## 1 为什么要开这门课——从零构建

> ⚠️ **核心危机**：研究者正在与底层技术脱节。

Percy 开篇直指问题核心。八年前的研究者会自己实现和训练模型；六年前至少会下载 BERT 做 fine-tune；现在很多人**只会 prompt 闭源模型**。

这不是全盘坏事——抽象层让我们能做更多事，大量研究因此解锁。但要警惕：

> 🔑 **关键判断**：LLM 的抽象层是**泄漏的（leaky）**。不像操作系统或编程语言那样边界清晰，LLM 是"string in, string out"的黑箱。要从事基础研究，必须**拆开整个技术栈**，协同设计数据、系统和模型。

Tatsu 补充：他和 Percy 花了很久想"什么才是真正有深度的技术内容"，结论是——**你必须从零构建，才能真正理解它（build it from scratch to understand it）**。这是整门课的精神基调。

### 1.1 小模型能不能代表大模型？

这是本课的核心张力。我们只训得起小模型，但：

| 现象 | 小模型表现 | 大模型真实情况 |
|------|-----------|--------------|
| **FLOPs 分布** | Attention ≈ MLP（约 1:1） | 175B 时 MLP 占绝对主导 |
| **涌现能力** | 看不到 in-context learning | 某个 scale 后突然出现 |

> ⚠️ **陷阱**：如果你在小规模上死磕 Attention 优化，可能优化错了对象——大规模时它的占比已被 MLP 淹没。

但这不意味着小模型无用。Percy 定义了**三类知识**：

| 知识类型 | 能否教 | 说明 |
|---------|--------|------|
| **机制（Mechanics）** | ✅ 能 | Transformer 怎么算、模型并行怎么用——这是"原料" |
| **心态（Mindset）** | ✅ 能 | 极致压榨硬件、严肃对待 scaling——OpenAI 凭这个杀出来 |
| **直觉（Intuition）** | ⚠️ 部分能 | 哪些数据/架构决策有效——因为随 scale 变化，只能教一半 |

> 💡 **预期管理**："你能拿走 2.5 / 3，已经物超所值。"

### 1.2 Bitter Lesson 的正确读法

> ⚠️ **常见误读**：Bitter Lesson = scale 决定一切，算法不重要，砸钱就行。

> ✅ **正确读法**：**算法在 scale 下**才重要。模型质量 = 效率 × 资源。效率在大规模下**更重要**——烧一亿美元时你不能浪费。

实证：OpenAI 2020 年论文显示，2012-2019 年 ImageNet 训练的**算法效率提升 44×**，比摩尔定律还快。

> 🔑 **课程的核心问题**：给定固定的 compute + data 预算，你能训练出的**最好模型**是什么？这个问题在任何 scale 下都有意义。

---

## 2 LLM 简史——一切并非凭空发生

> 💡 **要点**：Transformer 的所有原料在 2020 年前已就位。OpenAI 的贡献是**scaling 心态 + 工程执行**。

| 年代 | 里程碑 |
|------|--------|
| ~1948 | Shannon 用语言模型估算英语熵 |
| 2003 | Bengio 团队首个**神经语言模型** |
| ~2014 | Seq2seq（Ilya 等）、Adam 优化器 |
| 2017 | **Attention Is All You Need**（Transformer） |
| 2018-2019 | MoE scaling、模型并行探索、100B 参数实验 |
| 2018-2020 | 基础模型时代：ELMo → BERT → T5 |
| 2020+ | GPT-2/GPT-3（OpenAI 拥抱 scaling laws） |
| 2022+ | 开放权重模型浪潮：Llama / DeepSeek / Qwen |

> 💡 **冷知识**：Google 2007 年就在 **2 万亿 token** 上训了 5-gram 模型——比 GPT-3 的 token 量还大。但因为不是神经网络，没有任何涌现能力。

### 2.1 开放性的三个层级

| 层级 | 例子 | 你能拿到什么 |
|------|------|------------|
| **Closed** | GPT-4 | 只能 API 调用，零细节 |
| **Open weight** | Llama | 权重 + 架构细节，无数据细节 |
| **Open source** | OLMo / DeepSeek（部分） | 权重 + 数据 + 诚实的技术报告 |

> 🔑 但无论多开放的论文，都**替代不了亲手构建**。

---

## 3 五大模块总览

> 💡 **全局视角**：给一个 Common Crawl dump + 32 块 H100 + 两周，你该怎么训模型？这五个模块是五类设计决策。

### 3.1 模块一：Basics（A1）

目标：跑通一条完整训练管线。

| 组件 | 你要实现的 | 关键决策 |
|------|-----------|---------|
| **Tokenizer** | BPE（从零） | 字符→整数 |
| **模型架构** | Transformer | 激活函数、位置编码、归一化 |
| **训练** | Cross-entropy + AdamW | 学习率、batch size |

> ⚠️ **Percy 警告**：BPE 是 A1 里最出人意料的重活。别低估。

Transformer 自 2017 年来有很多"小改进"，累积起来差异巨大：

- **激活函数**：ReLU → SwiGLU（noam 论文原话："we offer no explanation except divine benevolence"）
- **位置编码**：绝对 → RoPE（Rotary）
- **归一化**：LayerNorm → RMSNorm；放置位置从 post-norm → pre-norm
- **MLP**：dense → Mixture of Experts
- **Attention**：full → sliding window / linear / GQA / MLA
- **替代架构**：SSM（Hyena），或 hybrid

> 💡 **assignment 细节**：用 TinyStories + OpenWebText 训练，leaderboard = 90 分钟 H100 内最小化 OpenWebText perplexity。

### 3.2 模块二：Systems（A2）

> 🔑 **核心目标**：从硬件中榨取最大性能。

**Kernel（单 GPU）**：GPU = 庞大的浮点运算阵列 + 分层内存（L2/L1 cache on-chip, HBM off-chip）。

> 💡 **仓库-工厂类比**：内存是仓库，计算是工厂，瓶颈是**搬运成本（data movement）**。目标：用 fusion、tiling 等技术最小化搬运。

**Parallelism（多 GPU）**：8 卡时 NVLink/NVSwitch 互连，GPU 间数据搬运更慢。策略：
- Data parallelism（数据并行）
- Tensor parallelism（张量并行，如 FSDP）

**Inference**：两阶段——
| 阶段 | 特征 | 瓶颈 |
|------|------|------|
| **Prefill** | prompt 一次性灌入，天然并行 | compute-bound |
| **Decode** | 自回归逐 token 生成 | memory-bound，难饱和 GPU |

> 💡 **经济现实**：推理成本正超越训练成本——训练是一次性的，推理随用量线性增长。

加速推理的手段：用更便宜的模型 / speculative decoding（小模型探路、大模型批量验收）。

### 3.3 模块三：Scaling Laws（A3）

> 🔑 **核心问题**：给你一个 FLOPs 预算，模型该多大？

- 大模型 = 少数据；小模型 = 多数据。最佳配比是什么？
- **Chinchilla 最优**：每个参数约配 **20 个 token**。
- 例：1.4B 模型应训 28B token。

> ⚠️ 但 Chinchilla 只优化训练 loss，**不考虑推理成本**。

> 💡 **A3 玩法**：课程提供一个 training API，你用超参查询 loss，拟合 scaling law，在 FLOPs 预算内预测最优 (params, data) 组合。用完了预算就没了——模拟前沿实验室的决策压力。

### 3.4 模块四：Data（A4）

> ⚠️ **澄清**："我们在互联网上训模型"——这句话是错的。数据不会从天而降，必须**主动获取**。

Percy 现场展示 Common Crawl 随机 10 篇：大量是垃圾、spam、非目标语言。**网络比你想的脏得多。**

数据管线：

```
crawl (HTML/PDF/code dirs)
  → HTML→text 转换（有损！）
  → 质量过滤（训分类器）
  → 去重（deduplication）
  → 有害内容移除
  → 可喂训练的文本
```

> 💡 前沿模型现在**花钱买数据**——公开网络的数据已不足以支撑极致性能。

### 3.5 模块五：Alignment（A5）

Base model 只会续写下一个 token。Alignment 让它**有用**，三个维度：

| 维度 | 目标 |
|------|------|
| **指令遵循** | 真正执行指令，而非续写指令文本 |
| **风格控制** | 长短、格式、语气 |
| **安全** | 拒绝有害请求 |

**两阶段**：

1. **SFT（Supervised Fine-Tuning）**：收集 (user, assistant) 对，监督学习。
   > 💡 LIMA 证明：1000 条高质量样本就够让 base model 获得指令遵循能力——因为 base model 已有"原始潜力"。

2. **学习反馈**：标注更轻量，算法做更多。
   - **偏好数据**：生成 A/B，人选更好的。
   - **验证器**：数学/代码有形式验证；或训一个 reward model。
   - **算法**：PPO → DPO（只需偏好数据，更简单）→ GRPO（DeepSeek，去 value function 的 PPO 变体）。

---

## 4 效率作为统一视角

> 🔑 **贯穿五模块的主线**：我们在 **compute-constrained** regime。所有设计决策都源于"压榨硬件"。

| 模块 | 效率驱动的设计决策 |
|------|------------------|
| 数据 | 激进过滤——不把算力浪费在坏数据上 |
| 分词 | 字节级优雅但算力浪费，BPE 是效率妥协 |
| 架构 | 大量决策由计算效率驱动 |
| 训练 | 单 epoch——我们赶时间，要多见数据而非多看单点 |
| Scaling laws | 用更少算力找最优超参 |
| 对齐 | 算力投入对齐 → 可用更小 base model |

> 💡 **趋势转变**：前沿实验室正从 compute-constrained 走向 **data-constrained**。到那时，单 epoch 不再合理，甚至架构都可能改变（Transformer 本身就是 compute efficiency 的产物）。

---

## 5 分词深入——BPE

> 📺 参考：Andrej Karpathy 的分词视频——本课许多"从零构建"理念受他启发。

### 5.1 分词是什么

**Tokenizer** = 字符串 ↔ 整数序列的双向映射。

- 每个整数 = 一个 **token**
- **词汇表大小（vocab size）** = token 能取的整数值范围

> 💡 用 [tiktokenizer.io](https://tiktokenizer.io) 可以直观玩各种现成 tokenizer。

**观察（GPT-4o tokenizer）**：
- 空格**属于** token 的一部分（`" hello"` ≠ `"hello"`，是两个不同 token）
- 空格按惯例**前缀**到 token（pre-tokenizer 的产物）
- 数字被切成任意片段（左到右，不按千位分组）

### 5.2 四种方案的演进

#### 方案 1：字符级（Character-based）

每个 Unicode 字符 → 码点（code point）整数。`a` → 97，`🌍` → 127757。

> ⚠️ **问题**：
> - 压缩比差（每个 token 只代表少量字节）
> - 词汇表巨大（149k+ Unicode 码点）
> - 稀疏：有些字符极罕见，白白占 vocab 名额

#### 方案 2：字节级（Byte-based）

一切转 UTF-8 字节。词汇表固定 0-255。

> ✅ 优点：vocab 小、无稀疏
> ❌ **致命缺点**：压缩比 = 1.0（1 byte/token）。序列太长 → attention 是 O(N²) 的，效率灾难。

> 💡 Percy 心声："我真希望 byte-level 能行，它最优雅——但今天的架构下它太慢了。"

#### 方案 3：词级（Word-based）

用正则分词（如 GPT-2 的 pre-tokenizer regex），每个 segment = 一个 token。

> ❌ **致命缺点**：词汇表**无界**。新词不断出现，全得映射到 `<UNK>`。perplexity 计算会因此失真。

#### 方案 4：BPE（Byte Pair Encoding）✅

> 🔑 **核心洞察**：不要预设怎么切——**在原始文本上训练 tokenizer**，让频繁序列合并成一个 token，罕见序列拆成多个 token。

**历史**：Philip Gage 1994 年为数据压缩发明；2015 年引入 NMT（Sennrich 等）；GPT-2 正式带入语言建模。

### 5.3 BPE 算法详解

**预处理**：先按 word-based 方式把文本切成 segments（GPT-2 的 pre-tokenizer），再对每个 segment 独立跑 BPE。

**训练算法**（以 `"cat and hat"` 为例）：

```
输入: 文本的字节序列
目标: 执行 num_merges 次合并
```

**第 1 轮**：
1. 把字符串转成字节序列（如 `[99, 97, 116, 32, 97, 110, 100, 32, 104, 97, 116]`）
2. 统计所有**相邻字节对**的频次
3. 找出频次最高的一对（如 `t,h` → `[116, 104]`，出现 2 次）
4. 分配新 vocab 槽位：`256`（因为 0-255 已被原始字节占用）
5. 在序列中把 `[116, 104]` 替换成 `256`

**第 2 轮**：
1. 在新序列上重新统计相邻 token 对
2. 找到新的最高频对（如 `256, 101` → 已合并的 `th` + `e`）
3. 分配 vocab `257`
4. 替换

**重复**直到完成 `num_merges` 次。

> 💡 **关键性质**：每轮序列长度**缩短**——压缩比越来越好，vocab 越来越大。

### 5.4 Encode 与 Decode

**Encode（推理时用）**：
```
输入: 新字符串
步骤: 转字节 → 按训练时 merges 的顺序，依次应用每个 merge
输出: 整数序列
```

> ⚠️ Percy 提示的优化点：朴素实现遍历所有 merges，但应该只遍历**相关的** merges。这是 A1 性能优化点之一。

**Decode**：
```
输入: 整数序列
步骤: 反查 vocab（整数 → 字节）→ 拼接字节 → UTF-8 解码
输出: 原始字符串
```

> ✅ **可逆性**：tokenization 必须是往返一致（round-trip）的，这是设计约束。

### 5.5 为什么 BPE 赢了

| 维度 | Character | Byte | Word | **BPE** |
|------|-----------|------|------|---------|
| Vocab 大小 | 巨大 | 256（太小） | 无界 | **可控** |
| 压缩比 | 差 | 1.0（差） | 好 | **好（自适应）** |
| 稀疏问题 | 严重 | 无 | 严重（UNK） | **无** |
| 可逆 | ✅ | ✅ | ❌（UNK） | **✅** |
| 自适应 | ❌ | ❌ | ❌ | **✅（看语料统计）** |

> 💡 Percy 结语："我希望有一天不用再讲这堂课——因为那时我们有直接处理字节的架构了。在那之前，我们还得跟分词打交道。"

---

## 6 总结与延伸

### 核心论点

1. **从零构建是理解的前提**：抽象层是 leaky 的，基础研究必须拆栈。
2. **效率是统一主线**：compute-constrained regime 下，每个决策都是效率驱动。
3. **小模型学机制和心态，直觉只能学一半**：但 2.5/3 已物超所值。
4. **BPE 是当前的效率妥协**：优雅的字节级方案在当前架构下太慢。

### 下节预告

Lecture 2：PyTorch 构建块 + **Resource Accounting**——精确追踪每一份 FLOPs 去了哪里。

---

## 📖 配套阅读（已在 Readwise，tag `cs336/a1`）

| 论文 | 对应知识点 |
|------|-----------|
| Attention Is All You Need (Vaswani 2017) | Transformer 基础 |
| RoFormer: Rotary Position Embedding (Su 2021) | RoPE |
| Layer Normalization (Ba 2016) | LayerNorm → RMSNorm |
| Adam (Kingma 2014) | 优化器 |
| Decoupled Weight Decay / AdamW (Loshchilov 2017) | AdamW |
| TinyStories (Eldan 2023) | A1 训练数据 |

---

*本讲义由 ZCode 基于 CS336 L1 字幕 + 领域知识生成（2026-07-13）。无配图版——网络恢复后可补视频帧。*
