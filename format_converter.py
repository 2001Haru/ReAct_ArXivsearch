import json
import re

def convert_to_sharegpt(raw_text, system_prompt):
    # 提取问题 
    question_match = re.search(r"(?:Question|User):\s*(.+?)\n", raw_text)
    if not question_match:
        return None
    question = question_match.group(1).strip()

    # 准备对话列表
    conversations = []
    
    # 将文本按照 Thought/Action/Observation/Final Answer 拆分
    # 找出所有的推理块和工具返回块
    # 这里的正则捕捉 Thought + Action 作为一个 AI 回合，以及 Observation 作为一个 Tool 回合
    
    # 提取所有的步进块
    steps = re.split(r"(Observation:)", raw_text)
    
    # 第一回合：问题
    conversations.append({"from": "human", "value": question})

    # 处理中间逻辑
    # steps[0] 包含 Question 和第一组 Thought+Action
    first_gpt_part = re.search(r"(Thought:.*?Action:.*?)\n", steps[0], re.DOTALL)
    if first_gpt_part:
        conversations.append({"from": "gpt", "value": first_gpt_part.group(1).strip()})

    # 循环解析后续的 Observation 和随后的 Thought+Action
    for i in range(1, len(steps), 2):
        if steps[i] == "Observation:":
            # 获取 Observation 的内容（直到下一个 Thought 或 Final Answer）
            obs_and_next = steps[i+1]
            obs_content = re.split(r"(Thought:|Final Answer:)", obs_and_next)[0].strip()
            conversations.append({"from": "observation", "value": obs_content})
            
            # 获取接下来的 Thought+Action 或 Final Answer
            remaining = obs_and_next[len(obs_content):]
            if "Final Answer:" in remaining:
                final_content = remaining.strip()
                conversations.append({"from": "gpt", "value": final_content})
                break
            else:
                next_gpt_match = re.search(r"(Thought:.*?Action:.*?)\n", remaining, re.DOTALL)
                if next_gpt_match:
                    conversations.append({"from": "gpt", "value": next_gpt_match.group(1).strip()})

    return {
        "conversations": conversations,
        "system": system_prompt
    }

# 使用示例
if __name__ == "__main__":
    # 从你的 evaluation.py 中获取系统提示词
    system_prompt = "你是一个专业的学术新闻助理。你必须通过工具获取最新信息，禁止凭空猜测..."
    
    # 这里的 raw_output 是 DeepSeek 打印出来的完整长字符串
    raw_output = """Question: 过去一个月 AMD 在 GPU 架构方面有什么新研究？... (此处省略你提供的完整文本)"""
    
    data_item = convert_to_sharegpt(raw_output, system_prompt)
    
    # 保存为 LLaMA-Factory 格式的 JSON
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump([data_item], f, ensure_ascii=False, indent=2)
    
    print("转换完成!")