import json
import time
from openai import OpenAI

DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

INPUT_FILE = "TestData.json"
OUTPUT_FILE = "searchability_oracle.json"

ORACLE_PROMPT = """你是一名学术文献检索问题判断专家。
现在有一个一个ArXiv论文库检索任务：模型允许调用工具search_arxiv来进行论文检索，其需要通过多轮 Thought-Action-Observation 来找到匹配用户问题的正确相关信息，最终输出一个Final Answer。
而这些问题有些实际上无法被检索到强相关信息。因此我们制作了这些【标准参考推理轨迹】来判断某个问题是否可以被"检索到"。你的任务就是根据提供的【标准参考推理轨迹】来判定该问题在 ArXiv 论文库中是否“可检索”。

判定逻辑：
你可以重点聚焦推理轨迹最后的 Final Answer 部分内容。如果 Final Answer 对于用户问题给出了详实有效的信息，判定该问题为 "searchable"。
而如果轨迹在多次更换关键词搜索后，Final Answer中指出是“未找到极其有效信息”或“ArXiv上不存在”或“建议去其他平台查看”等等，判定该问题为 "unsearchable"。

注意：轨迹可能在承认无法找到强相关信息之后给出一些模糊的建议或者弱相关的"可能有用"信息，这些都不算作“检索到”有效信息，请务必以“未找到强相关信息”作为判定依据。如果你仍没有确切把握，建议你考察轨迹中的其他内容辅助判断。

请直接返回 JSON 格式：{"question": "...", "status": "searchable/unsearchable"}"""

def get_searchability(question, trajectory):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": ORACLE_PROMPT},
                {"role": "user", "content": f"问题：{question}\n轨迹内容：{trajectory}"}
            ],
            response_format={'type': 'json_object'}
        )
        return json.loads(response.choices[0].message.content).get("status")
    except Exception as e:
        print(f"请求失败: {e}")
        return "searchable" # 默认值

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    oracle_map = {}
    for i, item in enumerate(data):
        question = item['conversations'][0]['value']
        # 将轨迹拼接为文本
        traj_text = ""
        for conv in item['conversations']:
            traj_text += f"{conv['from']}: {conv['value']}\n"
        
        print(f"[{i+1}/{len(data)}] 正在判定: {question[:20]}...")
        status = get_searchability(question, traj_text)
        oracle_map[question] = status
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(oracle_map, f, ensure_ascii=False, indent=2)
    print(f"\n黄金准则已生成至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()