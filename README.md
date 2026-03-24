# README 

## 中文版

### 项目说明

这是本项目的数据与代码自述文件。  
本项目的核心 feature 是：🔍 探究 ReAct 范式对小规模语言模型在检索交互任务中的推理性能提升效果。

> ⭐ 研究重点：小模型 检索交互 ReAct 推理范式
>
> 📘 如果您希望更细致地了解项目目标、实验意图与执行过程，建议阅读文档中的 Project Report，其中详细记录了我们的意图与行动。

### 文件功能索引

#### `Project Report`
📈 项目最终核心的的探究成果。
详细记载了整个项目的探究历程和方法，以及最终的结论。

#### `Data`
📦 项目所有 raw data 的集合，包括：
- 微调数据集
- 最终性能评测对应的推理轨迹
- 报告中使用的部分图表

您可以据此对实验数据与报告结论进行交叉核对，以验证数据真实性。✅

#### `LLM_as_Judge`
⚖️ 模型推理轨迹性能评测文件。  
我们使用 DeepSeek 作为更大容量的大模型，对待评测轨迹进行判定。  
分类式评价标准可在提示词中查看。

#### `arxivSearch`
🧰 任务主要调用工具。  
支持在特定时间范围内，按关键词检索最相关的一定数量结果。

#### `TestData_generate` 与 `TrainData_generate`
🧪 微调训练集与测试集标准答案生成模块。  
数据生成流程为：先由 DeepSeek 进行初步生成，再进行人工清洗，确保数据质量与可用性。

#### `evaluation`
📈 面向测试数据集生成模型推理轨迹的评测执行文件。

#### `format_converter`
🔄 将微调数据转换为 ShareGPT 格式所需的格式转换器。

---

## English Version

### Project Overview

This file is the data-and-code self-description README for the project.  
The core feature of this project is: 🔍 investigating how the ReAct paradigm (also often written as React in informal contexts) improves small language model reasoning performance in retrieval-interaction tasks.

> ⭐ Research focus: small models retrieval interaction ReAct reasoning paradigm
>
> 📘 If you want a more detailed understanding of the project goals and workflow, please read the Project Report, which documents our intentions and actions in detail.

### File Function Index

#### `Project Report`
📈 The main core product of this project.
详细记载了整个项目的探究历程和方法，以及最终的结论。
The report records in detail the entire project's exploration process and methods, as well as the final conclusions.

#### `Data`
📦 A collection of all raw data in this project, including:
- Fine-tuning datasets
- Reasoning trajectories used for final performance evaluation
- Some charts used in the report

This allows cross-checking between experimental data and reported conclusions to validate data authenticity. ✅

#### `LLM_as_Judge`
⚖️ Evaluation module for model reasoning trajectory quality.  
We use DeepSeek, a higher-capacity large model, to judge target trajectories.  
You can find our classification-style evaluation criteria directly in the prompts.

#### `arxivSearch`
🧰 Main tool used by the task pipeline.  
It retrieves a specified number of the most relevant results for given keywords within a specific time range.

#### `TestData_generate` and `TrainData_generate`
🧪 Standard-answer generation modules for fine-tuning and test datasets.  
Our data generation workflow is: initial generation with DeepSeek, followed by manual data cleaning.

#### `evaluation`
📈 Evaluation execution module for generating model reasoning trajectories on the test dataset.

#### `format_converter`
🔄 Format converter used to transform fine-tuning data into ShareGPT format.