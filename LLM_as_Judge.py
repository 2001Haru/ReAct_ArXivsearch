import json
import time
import re
from openai import OpenAI

# --- 配置区 ---
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  
MODEL_NAME = "deepseek-chat"  
STANDARD_DATA_FILE = "TestData.json"      # 作为 Oracle 的标准答案文件

INPUT_DATA_LIST = [
    "Baseline_nonreact.json",
    "Baseline_react.json",
    "Finetuning_nonreact.json",
    "Finetuning_react.json",
]
OUTPUT_EVAL_LIST = [
    "eval_Baseline_nonreact.json",
    "eval_Baseline_react.json",
    "eval_Finetuning_nonreact.json",
    "eval_Finetuning_react.json",
]

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def format_conversations_to_string(conversations):
    """
    将 ShareGPT 格式的对话列表转换为裁判可读的文本块
    """
    result = ""
    for conv in conversations:
        role = conv['from'].upper()
        content = conv['value']
        result += f"### {role}:\n{content}\n\n"
    return result

def get_judge_feedback(question, standard_traj, candidate_traj, preset_status):
    JUDGE_SYSTEM_PROMPT = """你是一名专业的 AI 智能体评测专家。你的任务是根据提供的【标准参考推理链】来评估【待评价模型推理轨迹】的表现。请严格按照以下评估维度与评分标准进行分类和打分，并给出简要评语。

### 评估维度与分类：
这些推理轨迹是均是一个ArXiv论文库检索任务的产物。待评估模型需要通过多轮 Thought-Action-Observation 来找到匹配用户问题的正确相关信息，最终输出一个Final Answer。待评估模型允许调用工具search_arxiv来进行论文检索。
我们人工撰写了【标准参考推理链】来作为每个问题的标准回答。由于ArXiv论文库内容有限，部分问题实际上无法被检索到强相关信息。系统会直接告诉你该问题是否可被检索到，你在此基础上评估待评价模型的表现。

### 输出要求：
你可以先做Analysis作为草稿，但是最终必须以 JSON 格式回复，包含用户问题是否可被检索，待评测轨迹的分类，评分以及简短评语。格式如下：
{
"classification": "...", 
"inform_score":"...",
"hallucination_score: "...",
"reason": "具体评语"
}

以下是待评测轨迹的各分类(classification)定义：
请注意，模型可能在输出Final Answer之后继续思考或者行动，你需要无视这之后的部分并且以第一个输出的Final Answer为重要评判对象！！(我们不允许作答后的补充)
1. 轨迹成功输出Final Answer(必须要严格看到明确的Final Answer字样！！):
    1.1. 问题可被检索类(Searchable)
        1.1.1. **完全正确 (Correct)**: 逻辑严密且信息源于 Observation。
        1.1.2. **偏差认知 (Deviation)**: 模型混淆了用户需求并且给出了不完全匹配的信息(比如用户要求12月模型搜索了全年)。
        1.1.3. **诚实无果 (Honest Failure)**: 模型未检索到关键信息，但是如实告知用户且未编造。
        1.1.4. **带幻觉推理 (Hallucination)**: 逻辑符合常理但 Thought 中包含 Observation 里没有的凭空编造信息(有些信息可能是正确的且来自模型常识，你需要仔细甄别是否属于幻觉)。

    1.2. 问题不可被检索类(Unsearchable)
        1.2.1. **青出于蓝 (Beyond Success)**: 逻辑严密且模型找到了标准推理轨迹没有检索到的关键信息(这种情况可能很少)。
        1.2.2. **偏差认知 (Deviation)**: 模型混淆了用户需求并且给出了不完全匹配的信息。
        1.2.3. **诚实说明 (Honest Success)**: 逻辑严密且在多次搜索无果后诚实承认无法检索到强相关内容。
        1.2.4. **带幻觉推理 (Hallucination)**: 逻辑符合常理但 Thought 中包含 Observation 里没有的凭空编造信息。

2. 轨迹未成功输出Final Answer(未能看到明确的Final Answer。):
    2.1. **完全失败 (Complete Failure)**: 模型全程甚至没有成功调用任何一次工具(你没有在推理轨迹中看到模型使用search_arxiv)
    2.2. **中途截止 (Interruption)**: 模型在某次调用工具或者思考之后截然而止，没有继续行动或者产生幻觉性等待。
    2.3. **幻觉重复 (Looping)**: 模型一直重复某段思考或者某个行动，导致最终推理步数达到预设上限而没有输出最终答案。

为了进一步帮助你进行打分，我们还会直接告诉你模型待评测轨迹是否输出了Final Answer以及是否调用过工具，因此你只需要做更细化的判断。

以下是对待评测轨迹的评分标准:
你需要给出两个评分：信息检索能力评分(inform_score)与幻觉评分(hallucination_score)，评分标准如下：
-信息检索评分：
若问题本身不可检索：-1分（请注意，无论模型是否输出Final Answer，只要问题不可检索，请都记为-1！！）
若问题是可检索的：
0分：模型未输出Final Answer。
0.5分：模型输出了Final Answer但是遗漏了标准答案的部分信息。
1分：模型输出了Final Answer且总结成果在信息全面度上达到或者超越标准答案水准。

-幻觉评分：
0分：模型未输出Final Answer。
0.5分：模型输出了Final Answer但是产生复读行为或者凭空编造了 Observation 中没有出现的信息(甚至编造Observation本身)。
1分：模型输出了Final Answer且没有明显幻觉。

在评语中，你需要简要说明模型推理情况并且给出分类和评分的理由。

### 示例：
【问题】: 为我总结NVIDIA GPU在2024年12月的研究进展。
【标准参考推理轨迹】: 
### gpt: Thought: 用户想了解2024年12月关于NVIDIA GPU的研究进展。我需要搜索2024年12月ArXiv上相关论文。关键词可以包括"NVIDIA GPU"等。时间范围设为2024年12月。\n\nAction: search_arxiv(\"NVIDIA GPU", \"20241201\", \"20241231\")"
### observation: 在指定的条件（日期: 20241201 至 20241231）下未找到相关学术文献。
...(此处省略部分过程)
### gpt: Thought: 多次的搜索均无果，这可能是因为ArXiv上确实没有相关信息，我需要向用户诚实说明这一点。\n\nFinal Answer: 多次搜索之后，并没有找到...
【待评价模型推理轨迹】:
### gpt: Thought: 我需要搜索2024年12月关于NVIDIA GPU的研究进展。\n\nAction: search_arxiv(\"NVIDIA GPU", \"20241201\", \"20241231\")"
### observation: 在指定的条件（日期: 20241201 至 20241231）下未找到相关学术文献。
...(此处再次省略部分过程)
### gpt: Thought: 我在多次搜索之后没有获得有效信息，我必须说明这可能是因为ArXiv确实不包含这些内容。\n\nFinal Answer: 我认为您所需要的信息无法在ArXiv论文库中找到...
### gpt: Thought: ...
经系统检测，待评价模型轨迹中调用过工具search_arxiv并且输出了Final Answer。此外，该问题本身属于不可被检索到(Unsearchable)。

Analysis: 我是DeepSeek，我需要公正地评判这条模型推理轨迹。模型成功输出了Final Answer但是之后又进行了思考，我需要忽视Final Answer之后的思考(这不影响分数和归类)。从标准参考推理轨迹来看，这个问题确实无法被检索到强相关内容，而模型诚实说明了这一点，并且没有观察到幻觉的产生。因此我可以开始归类和打分。
{
"classification":"1.2.3", 
"inform_score":"-1",
"hallucination_score: "1",
"reason": "在Final Answer中，模型面对不可检索问题诚实说明了情况，并且没有产生明显幻觉。"
}
"""

    existence_hint = judge_helper(candidate_traj)
    if preset_status == "unsearchable":
        existence_hint += "此外，该问题本身属于不可被检索到(Unsearchable)。"
    else:
        existence_hint += "此外，该问题本身是可被检索到(Searchable)。"

    user_content = f"【问题】: {question}\n\n【标准参考推理轨迹】:\n{standard_traj}\n\n【待评价模型推理轨迹】:\n{candidate_traj}\n\n{existence_hint}\n\n请根据上述信息，按照系统提示的格式和要求，给出你的评判结果。"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={'type': 'json_object'}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}
    
def judge_helper(candidate_traj):
    has_final = "Final Answer:" in candidate_traj
    has_action = "search_arxiv" in candidate_traj
    
    # 可以在 prompt 中明确告诉 DeepSeek 我们的物理检测结果
    if has_final:
        if has_action:
            existence_hint = "【系统检测】待评价模型轨迹中调用过工具search_arxiv并且输出了Final Answer。"
        else:
            existence_hint = "【系统检测】待评价模型轨迹中输出了Final Answer，但从未调用过工具search_arxiv。请注意模型幻觉行为。"
    else:
        if has_action:
            existence_hint = "【系统检测】待评价模型轨迹中调用过工具search_arxiv，但未输出Final Answer。"
        else:
            existence_hint = "【系统检测】待评价模型轨迹中从未调用过工具search_arxiv且未输出Final Answer。请注意模型可能的完全失败行为。"
    return existence_hint

def main():
    # 1. 加载数据
    for INPUT_DATA_FILE, OUTPUT_EVAL_FILE in zip(INPUT_DATA_LIST, OUTPUT_EVAL_LIST):
        with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        with open(STANDARD_DATA_FILE, 'r', encoding='utf-8') as f:
            standards_list = json.load(f)
        with open("searchability_oracle.json", 'r', encoding='utf-8') as f:
            oracle_data = json.load(f)

        # 2. 将标准答案建立索引，方便快速匹配
        # Key 是 human 的 value，Value 是转换后的长字符串
        standards_index = {}
        for item in standards_list:
            q = item['conversations'][0]['value']
            standards_index[q] = format_conversations_to_string(item['conversations'])

        eval_results = []

        # 3. 开始遍历评测
        for item in candidates:
            # 获取当前待测问题的 Question 内容
            question = item['conversations'][0]['value']
            candidate_traj = format_conversations_to_string(item['conversations'])
            
            # 匹配对应的标准答案
            standard_traj = standards_index.get(question)
            
            if not standard_traj:
                print(f"跳过问题 (未在 TestData 中找到标准答案): {question[:20]}...")
                continue

            preset_status = oracle_data.get(question, "searchable")

            print(f"正在评测: {question[:30]}...")
            judgment = get_judge_feedback(question, standard_traj, candidate_traj, preset_status)
            
            eval_results.append({
                "question": question,
                "judgment": judgment,
            })

        # 4. 保存结果
        with open(OUTPUT_EVAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()