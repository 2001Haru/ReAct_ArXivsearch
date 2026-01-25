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
Thought: 根据Observation，进行下一步思考。
...
Final Answer: 给出你的最终答案。

请基于工具返回的 Observation 进行回答，禁止凭空猜测。当你输出Final Answer之后结束对话。
强烈建议在至多4次查询后给出Final Answer。如果一直搜索无果，请简短化搜索关键词，或者诚实说明某时间段似乎没有所需结果，并且提供可能有用的信息。"""

class DeepSeekAgent:
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        """
        初始化 DeepSeek API 代理
        :param api_key: DeepSeek API Key
        :param base_url: DeepSeek 官方 API 地址
        """
        print(f"配置 DeepSeek API 接入")
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
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
    
    agent = DeepSeekAgent(API_KEY)
    
    # 测试问题
    questions = [
    # === 第一板块：GPU 架构演进与前沿趋势 (10题) ===
    "请分析 2024 年下半年关于 GPU 散热技术（如液冷、相变散热）的研究进展。",
    "为我总结 2024 年至 2025 年间 NVIDIA Blackwell 架构在学术论文中提及的核心技术特性。",
    "请调查 2024 年关于高带宽显存（HBM3e/HBM4）在 AI 加速器中应用的最新研究。",
    "我想了解 2025 年关于下一代 GPU 互连技术（如 NVLink 演进或光电互连）的学术评价。",
    "请总结 2024 年关于 GPU 硬件级显存池化与共享技术在算力集群中的优化方案。",
    "为我分析 2023-2024 年间 AMD GPU（RDNA/CDNA）在高性能计算领域的架构演进路径。",
    "请讲讲 2025 年关于 GPU 专用压缩引擎对提升大模型加载速度的相关研究。",
    "帮我汇总 2024 年关于 GPU 虚拟化与切片技术在云端 AI 算力分配中的最新成果。",
    "我想了解 2025 年关于 GPU 缓存一致性协议在大规模分布式训练中的性能损耗研究。",
    "请总结 2024 年关于 GPU 架构中 Tensor Core 针对低精度运算（FP8/FP4）的硬件支持现状。",

    # === 第二板块：国产芯片与异构计算生态 (10题) ===
    "为我总结华为 Ascend NPU 对于模型推理加速在 2025 年全年的研究成果。",
    "请讲讲 2024 年关于国产 GPU（如摩尔线程、天数智芯）在适配主流深度学习框架上的进展。",
    "为我说说 2024 年关于华为 CANN 架构在大模型分布式并行计算上的最新优化论文。",
    "我想了解 2025 年国产 AI 芯片在提升低精度运算（FP8/INT8）稳定性上的学术探索。",
    "请分析 2024 年关于华为昇腾集群在万卡规模下通信效率的实测与改进研究。",
    "为我总结 2025 年国产专用 AI 加速器（NPU）在端侧大模型推理上的能效表现研究。",
    "请分析 2024 年关于国产自主 GPU 指令集开发与编译器优化的学术应用现状。",
    "带我看看 2024-2025 年间国产芯片在先进封装（如 Chiplet, 3D 堆叠）上的技术进展。",#############
    "为我总结 2024 年关于国产异构计算平台在跨芯片负载均衡算法上的最新研究。",
    "请说说 2025 年关于国产 GPU 针对 DeepSeek 等高效开源模型的底层优化适配情况。",

    # === 第三板块：专用加速器与云端/边缘架构 (10题) ===
    "请分析 2024-2025 年间谷歌 TPU 系列（v5p/v6）在架构演进与扩展性上的研究。",
    "为我总结 2025 年关于亚马逊自研芯片 Trainium 系列在处理大规模模型训练时的性能表现。",
    "请讲讲 2024 年关于特斯拉 Dojo 超算芯片在自动驾驶端到端训练中的硬件优化研究。",
    "我想了解 2024 年关于 Groq LPU 架构及其 SRAM 方案在降低推理延迟方面的技术解析。",
    "请总结 2025 年关于微软 Maia 100 芯片在 Azure 云环境下的部署表现与评价。",
    "为我总结 2024 年关于苹果 M 系列芯片（M4/M5）在端侧 NPU 架构上的技术革新。",
    "请讲讲 2025 年关于高通骁龙系列芯片针对本地大模型推理的硬件加速机制研究。",
    "我想看看 2025 年上半年关于存算一体（CIM）芯片在边缘 AI 任务中的最新学术突破。",
    "请分析 2024 年关于专用推理芯片在处理极低比特（1-bit/2-bit）模型时的硬件潜力。",
    "为我总结 2024 年关于 AWS Inferentia 系列在处理跨区域分布式推理时的性能优化研究。",

    # === 第四板块：算法、精度与软硬协同优化 (10题) ===
    "请讲讲 2024-2025 年间大语言模型从 FP16 到低精度 FP4 转换的学术研究脉络。",
    "为我总结 2024 年关于混合精度训练在保持大模型收敛性方面的最新硬件优化策略。",
    "请分析 2025 年关于 1.58-bit 量化模型（如 BitNet）在现有硬件架构上的效率瓶颈研究。",
    "我想了解 2024 年关于 KV Cache 压缩技术对缓解 GPU 推理显存占用的实验对比研究。",
    "请说说 2025 年关于 FlashAttention 系列算法在顶级 GPU 硬件上实现性能加速的原理。",##########
    "为我总结 2024 年关于模型并行策略对不同硬件带宽（NVLink/PCIe）利用率影响的研究。",##########
    "请分析 2025 年关于混合专家模型（MoE）负载均衡算法在硬件层面的优化成果。",
    "为我说说 2024 年关于大模型推理中推测性解码对硬件算力配比要求的最新论文。",
    "请分析 2025 年关于神经网络剪枝与稀疏化技术在新型 GPU 架构上的加速实测表现。",
    "为我总结 2025 年关于 AI 编译器（如 Triton, MLIR）在屏蔽底层芯片架构差异上的最新进展。",##########

    # === 第五板块：基础设施、能源与前沿交叉 (10题) ===
    "请讲讲 2024-2026 年间 AI 数据中心采用新型能源供电对硬件稳定性的评价研究。",
    "为我总结 2025 年关于液冷技术在大规模算力集群中降低 PUE 值的学术调研成果。",
    "请分析 2024 年关于数据中心互联中光电共封（CPO）技术对算力扩展性的贡献研究。",
    "我想了解 2025 年关于量子计算模拟器在 NVIDIA GPU 集群上运行效率的最新评估。",
    "为我总结 2024 年关于 AI 集群在面对电力波动时的硬件级电压频率保护策略。",
    "请分析 2025 年末关于全球显存产能瓶颈对未来 AI 芯片技术演进方向的学术影响。",##########
    "为我总结 2024 年关于 AI 训练过程中的碳足迹追踪与硬件级低功耗模式的研究进展。",
    "请分析 2024 年关于计算存储（CSD）在处理大规模训练语料预处理中的加速效果研究。",
    "帮我梳理 2024-2025 年关于分布式算力网络中跨地域节点性能对比的学术调研。",
    "为我总结 2026 年初学术论文中提及的关于“自适应 AI 硬件架构”的最新设想。"
]
    
    # DeepSeek对于被标记问题的回答不好，重新回答
    comp_questions = [
        "带我看看 2024-2025 年间国产芯片在先进封装（如 Chiplet, 3D 堆叠）上的技术进展。",
        "请说说 2025 年关于 FlashAttention 系列算法在顶级 GPU 硬件上实现性能加速的原理。",
        "为我总结 2024 年关于模型并行策略对不同硬件带宽（NVLink/PCIe）利用率影响的研究。",
        "为我总结 2025 年关于 AI 编译器（如 Triton, MLIR）在屏蔽底层芯片架构差异上的最新进展。",
        "请分析 2025 年末关于全球显存产能瓶颈对未来 AI 芯片技术演进方向的学术影响。",
    ]
    
    all_data = []

    for question in comp_questions:
        print(f"开始处理问题: {question}")
        result_log = agent.run_test(question)
        data_item = convert_to_sharegpt(result_log, NAIVE_SYSTEM_PROMPT)

        if data_item:
            all_data.append(data_item)
        
     # 保存为 LLaMA-Factory 格式的 JSON
    with open("TestData_comp.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)