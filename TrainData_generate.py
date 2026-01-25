import re
import time
import datetime
from openai import OpenAI
from arxivSearch import search_arxiv_tool
from datetime import date
from format_converter import convert_to_sharegpt
import json

CURRENT_DATE = str(date.today().strftime('%Y-%m-%d'))  # 当前时间

NAIVE_SYSTEM_PROMPT = f"""你是一个专业的学术新闻助理。现在日期是 {CURRENT_DATE}。
你可以通过以下工具获取ArXiv平台最新学术信息, 请根据用户问题合理选择搜索的时间范围。

### 工具说明：
- search_arxiv(query, start_date, end_date): 
  - query: 搜索关键词。
  - start_date: 开始日期 (格式: YYYYMMDD)。
  - end_date: 结束日期 (格式: YYYYMMDD)。
### 使用示例: search_arxiv("NVIDIA GPU", "20240101", "20241231")

### 输出格式建议：
Thought: 简要描述你的思考。
Action: search_arxiv("关键词", "开始日期", "结束日期")。一旦输出Action，等待Observation反馈即可。
Observation: 工具返回的结果摘要。(你绝对不允许自己凭空编写Observation!)
...
Final Answer: 给出你的最终答案。

请基于工具返回的 Observation 进行回答，禁止凭空猜测。当你输出Final Answer之后结束对话。
强烈建议在至多4次查询后给出最终答案。如果一直搜索无果，请精简搜索关键词，或者诚实说明某时间段似乎没有所需结果，并且提供可能有用的信息。"""

class DeepSeekAgent:
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        """
        初始化 DeepSeek API 代理
        :param api_key: 你的 DeepSeek API Key
        :param base_url: DeepSeek 官方 API 地址
        """
        print(f"正在配置 DeepSeek API 接入...")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = "deepseek-chat" # 或者使用 "deepseek-reasoner" 如果需要更强的推理能力
        self.max_steps = 6  # 限制最大交互步数

    def llm_generate(self, messages):
        """
        调用 DeepSeek API 生成内容，包含重试逻辑
        """
        max_retries = 5
        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,  # 评估通常需要较低的随机性
                    stop=["Observation"],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                wait_time = 2 ** i
                print(f"API 调用出错: {e}, {wait_time}s 后重试...")
                time.sleep(wait_time)
        
        return "Error: 达到最大重试次数，无法获取模型响应。"

    def run_test(self, user_query):
        # 初始化对话历史
        messages = [
            {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
        
        trajectory = f"Question: {user_query}\n"
        step = 0
        
        while step < self.max_steps:
            # 获取模型输出
            response = self.llm_generate(messages)
            trajectory += response + "\n"
            
            # 将模型的回答加入对话上下文
            messages.append({"role": "assistant", "content": response})
            
            # 如果模型给出了最终答案，停止测试
            if "Final Answer" in response:
                break

            # 使用正则解析工具调用语句
            action_match = re.search(
                r"search_arxiv\(['\"](.+?)['\"]\s*,\s*['\"](\d{8})['\"]\s*,\s*['\"](\d{8})['\"]\s*\)", 
                response
            )
            
            if action_match:
                q, s, e = action_match.groups()
                print(f"[*] 执行工具调用: search_arxiv('{q}', '{s}', '{e}')")
                
                # 获取工具观察结果
                observation = search_arxiv_tool(q, start_date=s, end_date=e)
                obs_text = f"{observation}"
                
                trajectory += obs_text + "\n"
                # 将观察结果作为系统/用户反馈加入上下文
                messages.append({"role": "user", "content": obs_text})
                
                step += 1
            else:
                # 如果没有 Action 也没有 Final Answer，尝试提醒模型或跳出
                print("提示：模型未按照标准格式输出 Action 或 Final Answer。")
                break
        
        return trajectory

if __name__ == "__main__":
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
    
    agent = DeepSeekAgent(API_KEY)
    
    # 测试问题
    questions = [
    # === 第一板块：GPU 架构演进与技术趋势 (30题) ===
    "请讲讲 2024 年 NVIDIA 在 Blackwell 架构设计中展现出的那些核心技术改进趋势。",
    "能为我总结一下 2023 年至 2024 年间 AMD GPU 在 RDNA 与 CDNA 架构上的融合发展路径吗？",
    "我想了解 2025 年学术界对于下一代 GPU 采用 HBM4 内存的技术预测与主要挑战。",
    "带我看看 2024 年关于 GPU 硬件级异步计算（Asynchronous Compute）优化的主要论文成果。",
    "为我说说 2022 年以来 NVIDIA Hopper 架构在高性能计算（HPC）领域的实测性能评估情况。",
    "请总结一下 2025 年上半年关于 GPU 显存池化（Memory Pooling）技术的最新研究进展。",
    "帮我对比分析 2024 年关于 GPU 互连技术（如 NVLink vs. PCIe 6.0）在扩展性上的学术评价。",
    "我想知道 2023 年关于多模态模型在主流 GPU 上的内存布局优化有哪些研究成果。",
    "为我详细讲讲 2025 年关于 Blackwell 架构在处理长文本上下文（Long Context）时的显存管理策略。",
    "请分析一下 2024 年下半年关于 GPU 散热技术（如液冷）对算力稳定性影响的研究报告。",
    "带我回顾 2024 年 NVIDIA GB200 系统架构在机架级能效比上的学术评价。",
    "为我总结 2022-2023 年间英特尔 Data Center GPU Max 系列在科学仿真中的应用成果。",
    "请讲讲 2025 年关于 GPU 专用解压引擎（Decompression Engine）对大模型加载加速的研究进展。",
    "帮我梳理 2024 年关于 GPU 架构中 Tensor Core 演进对低精度运算（FP8/FP4）的支持现状。",
    "为我说说 2023 年关于 AMD ROCm 软件生态在提升 GPU 训练兼容性方面的技术突破。",
    "请介绍 2025 年关于 GPU 算力集群中自愈技术（Self-healing）的最新学术论文。",
    "我想了解 2024 年关于 GPU 缓存一致性协议在大规模分布式训练中的性能损耗研究。",
    "带我深入看看 2022 年底 H100 发布后，学术界对其 Transformer Engine 实现逻辑的解析。",
    "请讲讲 2025 年关于光电混合互连在下一代 GPU 集群中降低延迟的实验数据。",
    "为我总结 2024 年关于 GPU 虚拟化与切片技术在 AI 云计算中的最新优化方案。",
    "帮我分析 2023 年关于特定领域加速器（DSA）与通用 GPU 在 Transformer 推理上的能效对比。",
    "请说说 2025 年关于 GPU 算力芯片在 2nm 工艺节点下的功耗密度挑战研究。",
    "带我看看 2024 年关于 GPU 显存压缩技术（Memory Compression）对大模型吞吐量的贡献。",
    "请讲讲 2022-2024 年间主流 GPU 在处理混合专家模型（MoE）时的带宽利用率变化。",
    "为我总结 2025 年关于 GPU 硬件级安全隔离（TEE）在隐私计算中的最新性能表现。",
    "我想知道 2024 年关于 GPU 指令集架构（ISA）演进对编译器自动化优化影响的研究。",
    "请说说 2023 年关于 AMD MI300 系列在异构内存管理上的技术创新评价。",
    "为我分析 2025 年关于 GPU 算力集群在处理超大规模（100T）模型时的网络拓扑优化。",
    "带我了解 2024 年关于 GPU 功耗预测模型（Power Modeling）在数据中心节能中的应用。",
    "请预测性地讲讲 2025 年末关于英伟达下一代 Rubin 架构在学术论文中提及的潜在特征。",

    # === 第二板块：国产芯片与异构计算生态 (25题) ===
    "请讲讲 2024 年华为昇腾（Ascend）系列芯片在适配主流深度学习框架上的主要进展。",
    "带我回顾 2023-2025 年间国产 GPU 厂商在解决硬件算力短板上的技术演进路线。",
    "为我说说 2024 年关于华为 CANN 架构在大模型分布式调度上的最新优化论文。",
    "我想了解 2025 年上半年国产 AI 芯片在提升 FP8 运算精度与稳定性上的学术探索。",
    "请为我总结 2024 年关于摩尔线程（Moore Threads）在 MUSA 架构上的软硬件适配成果。",
    "帮我讲讲 2023 年关于壁仞科技（Biren）GPU 在近内存计算（PIM）领域的代表性研究。",
    "请分析一下 2024 年关于华为昇腾集群在万卡规模下通信效率的实测与改进研究。",
    "为我总结 2025 年关于国产专用 AI 加速器（NPU）在端侧大模型推理上的能效表现。",
    "请说说 2024 年关于国产自主 GPU 指令集开发生态在学术界的应用现状。",
    "带我看看 2023-2025 年间国产芯片在 3D 堆叠封装技术上的突破对算力的影响。",
    "为我讲述 2024 年关于华为昇腾 910 系列在处理国产自研大模型时的性能反馈。",
    "请分析 2025 年关于国产异构计算平台在跨芯片负载均衡上的最新算法研究。",
    "帮我总结 2024 年关于寒武纪思元系列在处理特定视觉任务时的硬件加速优势。",
    "为我说说 2025 年上半年关于国产 GPU 编译器在算子自动融合技术上的最新论文。",
    "请讲讲 2024 年关于国产 AI 芯片在应对大规模显存缺口时的压缩技术应用情况。",
    "带我回顾 2023 年以来国产算力平台在开源社区（如 Hugging Face）的模型适配进展。",
    "为我对比 2024 年关于国产自研互连协议（如 HCCS）相比 NVLink 的技术差异。",
    "请讲讲 2025 年关于国产 NPU 在处理多模态长序列任务时的架构瓶颈研究。",
    "为我总结 2024 年关于国产芯片在政务与工业 AI 场景下的硬件选型对比研究。",
    "带我看看 2025 年初关于华为下一代昇腾 950 在学术论文中预测的技术参数。",
    "请分析 2024 年关于国产 GPU 针对量化模型（INT4/INT8）的硬件加速效率。",
    "为我讲讲 2023-2024 年间天数智芯在通用高性能计算任务中的稳定性评估结果。",
    "我想了解 2024 年关于国产算力芯片在构建国产算力网络中的节点性能对比。",
    "请说说 2025 年关于国产 GPU 针对 DeepSeek 等高效模型的底层优化适配情况。",
    "带我看看 2024 年关于国产 AI 芯片软件栈（如驱动、库）成熟度的学术调研。",

    # === 第三板块：专用加速器与云端/边缘架构 (20题) ===
    "请讲讲 2023-2025 年间谷歌 TPU 系列（v5p, v6）在架构演进上的主要逻辑。",
    "为我总结 2024 年关于谷歌 TPU v5p 在处理万亿级参数 MoE 模型时的扩展性研究。",
    "能带我分析 2025 年关于亚马逊自研芯片 Trainium2 在大规模模型训练中的能效表现吗？",
    "请讲讲 2024 年关于特斯拉 Dojo 超算芯片在自动驾驶端到端训练中的硬件优化。",
    "为我说说 2023 年关于 Groq LPU 架构采用 SRAM 方案对推理延迟影响的深度解析。",
    "我想了解 2024 年关于 Meta 自研推理芯片 MTIA 在推荐系统负载下的性能实测研究。",
    "请总结 2025 年关于微软 Maia 100 芯片在 Azure 云环境下的部署与性能反馈。",
    "带我回顾 2024 年关于苹果 M4 系列芯片在端侧 AI（NPU）架构上的技术革新。",
    "请讲讲 2025 年关于高通骁龙 8 Elite 针对本地大模型推理的硬件加速机制。",
    "为我总结 2024 年关于 AWS Inferentia2 在处理跨区域分布式推理时的时延优化。",
    "我想看看 2025 年上半年关于存算一体（CIM）芯片在边缘 AI 任务中的最新学术突破。",
    "请分析 2024 年关于 TPU v5e 在提升单位成本推理吞吐量方面的研究成果。",
    "为我讲讲 2025 年关于 RISC-V 架构在定制化 AI 张量加速器设计上的学术趋势。",
    "带我了解 2024 年关于移动端 SoC 中 NPU 与 GPU 协同处理 AI 任务的最新方案。",
    "请讲讲 2025 年关于专用推理芯片在处理极低比特模型（1-bit/2-bit）时的硬件潜力。",
    "为我总结 2024 年关于谷歌 Axion 处理器在 AI 预处理与逻辑调度中的实际贡献。",
    "请说说 2025 年关于基于 FPGA 的 AI 加速器在时延敏感任务中的表现。",
    "带我看看 2024 年关于 AI 专用服务器在数据中心内跨机架网络拥塞控制的研究。",
    "请分析 2025 年关于车载 AI 芯片在处理多路传感器输入时的硬件级并行策略。",
    "为我总结 2024 年关于微软/OpenAI 'Stargate' 超算项目中提及的硬件技术难点。",

    # === 第四板块：算法、精度与软硬协同优化 (15题) ===
    "请讲讲 2023-2025 年间大语言模型从 FP16 到 FP4 精度转换的学术研究脉络。",
    "为我总结 2024 年关于混合精度训练在保持模型收敛性方面的最新硬件策略。",
    "帮我分析 2025 年关于 1.58-bit 量化模型在现有 GPU 架构上运行的效率瓶颈。",
    "我想了解 2024 年关于 KV Cache 压缩技术对 GPU 推理显存占用缓解的实验对比。",
    "请说说 2025 年关于 FlashAttention-3 在顶级硬件上实现近物理极限加速的原理。",
    "带我回顾 2024 年关于模型并行策略对不同硬件带宽利用率影响的研究。",
    "请讲讲 2025 年关于混合专家模型（MoE）负载均衡算法在硬件层面的优化成果。",
    "为我总结 2024 年关于大模型推理中推测性解码对硬件算力的具体要求。",
    "请分析 2025 年关于神经网络剪枝与稀疏化在 Blackwell 等架构上的加速实测表现。",
    "为我说说 2024 年关于 Transformer 替代方案（如 Mamba）对 GPU 显存读写模式的影响。",
    "请讲讲 2025 年关于多令牌预测（MTP）架构对 AI 硬件内部传输带宽提出的挑战。",
    "带我看看 2024 年关于模型权重在 GPU-NPU 间进行无损传输与切换的最新研究。",
    "为我总结 2025 年关于 AI 编译器（如 Triton）在屏蔽底层架构差异上的最新进展。",
    "请说说 2024 年关于使用低比特量化加速向量数据库搜索的软硬协同方案。",
    "我想了解 2025 年关于多模态模型在异构集群中分配计算权重的最新调度算法。",

    # === 第五板块：基础设施、能源与未来前沿 (10题) ===
    "请讲讲 2024-2026 年间 AI 数据中心采用核能（SMR）供电对硬件稳定性的评价。",
    "为我总结 2025 年关于液冷技术在大规模算力集群中降低 PUE 值的学术调研成果。",
    "带我看看 2024 年关于数据中心互联中光电共封（CPO）技术对算力扩展的贡献。",
    "请分析 2025 年关于量子计算模拟器在 NVIDIA GPU 集群上运行效率的最新评估。",
    "为我总结 2024 年关于 AI 集群在面对极端高温环境下的硬件降额保护策略研究。",
    "请讲讲 2025 年关于全球 HBM 产能瓶颈对 2026 年 AI 芯片技术演进方向的影响。",
    "我想了解 2024 年关于 AI 训练过程中的碳足迹追踪与硬件级低功耗模式的研究。",
    "为我总结 2025 年末关于 3D 封装技术限制对高端 AI 芯片产出的学术分析。",
    "请分析 2024 年关于计算存储（CSD）在处理海量训练语料预处理中的加速效果。",
    "带我展望 2026 年初关于“AGI 硬件”概念在学术论文中提及的自适应架构构想。"
]
    
    all_data = []

    for question in questions:
        print(f"开始处理问题: {question}")
        result_log = agent.run_test(question)
        data_item = convert_to_sharegpt(result_log, NAIVE_SYSTEM_PROMPT)

        if data_item:
            all_data.append(data_item)
        
     # 保存为 LLaMA-Factory 格式的 JSON
    with open("data.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)