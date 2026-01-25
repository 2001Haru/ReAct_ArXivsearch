import re
import torch
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
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
请严格遵守以上搜索格式，错误格式(如search_arxiv(query=.., ..., ...)或search_arxiv.org(...))均是无效的!!!

### 输出格式建议：
Thought: 简要描述你的思考。
Action: search_arxiv("关键词", "开始日期", "结束日期")。一旦输出Action，等待Observation反馈即可。
Observation: 工具返回的结果摘要。
(观察到工具结果之后继续思考或行动，直到你有足够的信息给出最终答案)
...
Final Answer: 给出你的最终答案。(你必须输出Final Answer!)

请基于工具返回的 Observation 进行回答。当你输出Final Answer之后立即结束对话并且没有机会继续输出内容。
强烈建议在至多4次查询后给出最终答案。如果搜索无果，请简单化关键词继续搜索，或者诚实说明某时间段似乎没有所需结果，并且提供可能有用的信息。
以下是一条任务推理轨迹示例，你可以模仿这种格式进行更全面的回答：

#### 示例：
User: 请总结 2024 年NVIDIA公司对于GPU的研究进展。
Thought: 我需要了解 2024 年 NVIDIA 在 GPU 方面的研究进展，关键词可以是"NVIDIA GPU" "NVIDIA"。当前日期是2026年，我应该搜索2024年全年的资料。
Action: search_arxiv("NVIDIA GPU", "20240101", "20241231)

Observation: [1] (发布日期: 2024-06-21) 标题: Pulscan... 摘要: The Fourier Domain Acceleration Search (FDAS) and Fourier Domain Jerk Search (FDJS) ...
Thought: 我得知了关于Pulscan技术的一些研究内容。但是这些信息还不够全面，我可以更改搜索关键词以获得更多有用内容。
Action: search_arxiv("NVIDIA", "20240101", "20241231")

Observation: [1] (发布日期: 2024-11-15) 标题: Advances in NVIDIA... 摘要: Energy efficiency and AI workloads...
Thought: 现在我有足够的信息来总结 2024 年 NVIDIA 在 GPU 领域的研究进展。这一年NVIDIA聚焦于Pulscan技术和GPU架构的提升...
Final Answer: 2024 年，NVIDIA 在 GPU 领域取得了显著进展，特别是在 Pulscan 技术方面，该技术通过傅里叶域加速搜索和加速搜索显著提升了信号处理能力..."""

class MultiStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=["Observation:", "Observation", "\nObservation"]):
        self.tokenizer = tokenizer
        # 将所有停止词转换为 Token ID 列表
        self.stop_sequences = [
            tokenizer.encode(s, add_special_tokens=False) for s in stop_strings
        ]
        # 记录最长的一个停止词长度，用于优化截取范围
        self.max_len = max(len(seq) for seq in self.stop_sequences)

    def __call__(self, input_ids, scores, **kwargs):
        # 每次生成 token 后，检查 input_ids 的末尾
        for stop_seq in self.stop_sequences:
            target_len = len(stop_seq)
            if input_ids.shape[1] >= target_len:
                # 提取当前末尾对应长度的 token
                last_tokens = input_ids[0, -target_len:].tolist()
                if last_tokens == stop_seq:
                    return True
        return False
    
class StopOnCountCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, target_string='Final Answer', count=2):
        self.tokenizer = tokenizer
        self.target_string = target_string
        self.prompt_len = prompt_len
        self.count = count

    def __call__(self, input_ids, scores, **kwargs):
        # 将当前生成的所有 Token 解码为文本
        # 注意：这里我们只解码最近的一段，以提高效率
        new_generated_ids = input_ids[0][self.prompt_len:]
        generated_text = self.tokenizer.decode(new_generated_ids, skip_special_tokens=True)
        
        # 统计目标字符串出现的次数
        current_count = generated_text.count(self.target_string)
        
        # 如果达到了设定的次数 比如看到第 2 次 Final Answer，立刻返回 True 停止
        if current_count >= self.count:
            return True
        return False

class ComparisonAgent:
    def __init__(self, model_path, adapter_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 保持 4-bit 量化配置以节省显存
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 如果有适配器，则加载 LoRA 补丁
        if adapter_path:
            print(f"正在挂载 LoRA 适配器: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.has_adapter = True
        else:
            self.model = base_model
            self.has_adapter = False
        
        self.model.eval()

    def llm_generate(self, prompt, mode="sft"):
        """
        mode: "sft" 使用微调后的逻辑, "baseline" 禁用补丁回退到原始模型
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1] # 获取 Prompt 长度

        stop_words = ["Observation:", "\nObservation", "Observation\n", "Observation", "观察:","Action: 终止", "终止"]
        
        # 如果模型有适配器且用户要求使用 baseline 模式，暂时禁用 LoRA
        if self.has_adapter and mode == "baseline":
            # 使用 peft 的 disable_adapter 上下文管理器，不占额外显存地切回基准
            with self.model.disable_adapter():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.05,
                    stop_strings=stop_words,
                    tokenizer= self.tokenizer,
                    stopping_criteria=StoppingCriteriaList([MultiStopCriteria(self.tokenizer), \
                        StopOnCountCriteria(tokenizer=self.tokenizer, prompt_len=prompt_len)])  # 添加双重停止条件
                )
        else:
            # 正常生成
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.05,
                stop_strings=stop_words,
                tokenizer= self.tokenizer,
                stopping_criteria=StoppingCriteriaList([MultiStopCriteria(self.tokenizer), \
                        StopOnCountCriteria(tokenizer=self.tokenizer, prompt_len=prompt_len)])  # 添加双重停止条件
            )
            
        return self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def run_test(self, question, mode="sft"):
        # 将 mode 参数传递给推理流程
        trajectory = f"User: {question}\n"
        print(f"\n--- 正在以 [{mode.upper()}] 模式运行 ---")
        
        prompt = f"{NAIVE_SYSTEM_PROMPT}\nUser: {question}\nAssistant: "
        
        step = 0
        while step < 6:
            # 调用修改后的 generate 函数
            response = self.llm_generate(prompt, mode=mode)
            prompt += response
            trajectory += response + "\n"
            
            if "Final Answer" in response:
                break

            action_match = re.search(r"search_arxiv\(['\"](.+?)['\"]\s*,\s*['\"](\d{8})['\"]\s*,\s*['\"](\d{8})['\"]\s*\)", response)
            
            if action_match:
                q, s, e = action_match.groups()
                observation = search_arxiv_tool(q, start_date=s, end_date=e)
                obs_text = f"{observation}\n本次工具调用已结束。请继续思考或行动。\n"
                prompt += obs_text
                trajectory += obs_text
                step += 1
            else:
                break
        
        return trajectory
    

if __name__ == "__main__":
    MODEL_PATH = r"D:\HALcode\Models\qwen\Qwen3-1.7B" 

    ADAPTER_PATH = r"D:\HALcode\LLaMA-Factory\saves\Qwen3-1.7B-Thinking\lora\train_2026-01-18-17-23-18"
    
    #trained_agent = ComparisonAgent(MODEL_PATH, ADAPTER_PATH)
    baseline_agent = ComparisonAgent(MODEL_PATH)
    
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

    all_data = []
    all_data_raw = []

    for question in questions:
        print(f"开始处理问题: {question}")
        result_log = baseline_agent.run_test(question,mode="baseline")
        print(result_log)
        data_item_raw = result_log
        data_item = convert_to_sharegpt(result_log, NAIVE_SYSTEM_PROMPT)
        
        if data_item_raw:
            all_data_raw.append(data_item_raw)

        if data_item:
            all_data.append(data_item)
        
     # 保存为 LLaMA-Factory 格式的 JSON
    with open("Baseline_react.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

    with open("Baseline_react_raw.json", "w", encoding="utf-8") as f:
            json.dump(all_data_raw, f, ensure_ascii=False, indent=2)