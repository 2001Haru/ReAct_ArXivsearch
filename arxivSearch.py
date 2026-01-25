import requests
import xml.etree.ElementTree as ET

def search_arxiv_tool(query, start_date=None, end_date=None, max_results=6, mode='relevance',):
    """
    检索 ArXiv 数据库并支持日期筛选。
    
    :param query: 搜索关键词
    :param mode: 排序模式 ('relevance' 或 'submittedDate')
    :param start_date: 开始日期 (格式: 'YYYYMMDD', 例如 '20230101')
    :param end_date: 结束日期 (格式: 'YYYYMMDD', 例如 '20231231')
    :param max_results: 返回结果的最大数量
    """
    print(f"正在检索 ArXiv 数据库: {query}...")
    base_url = 'http://export.arxiv.org/api/query?'
    
    # 构造 search_query
    # ArXiv API 语法: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
    clean_query = query.strip().replace('"', '')
    # 使用括号和引号包裹，确保 all: 作用于整个短语
    search_parts = f'all:"{clean_query}"'
    
    if start_date or end_date:
        # 如果只提供了起始或结束日期，则补全另一端
        s = f"{start_date}0000" if start_date else "000001010000"
        e = f"{end_date}2359" if end_date else "999912312359"
        full_search_query = f'({search_parts}) AND submittedDate:[{s} TO {e}]'
    else:
        full_search_query = search_parts
    
    sort_by = 'submittedDate' if mode == 'submittedDate' else 'relevance'
    params = {
        'search_query': full_search_query,
        'start': 0,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': 'descending'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return f"Observation: 在指定的条件（日期: {start_date} 至 {end_date}）下未找到相关学术文献。"
        
        results = []
        for i, entry in enumerate(entries):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text[:10]
            link = entry.find('atom:id', ns).text
            
            results.append(
                f"[{i+1}] (发布日期: {published})\n"
                f"标题: {title}\n"
                f"摘要: {summary[:400]}..."
            )
        
        return "Observation: \n\n" + "\n\n".join(results)
        
    except Exception as e:
        return f"Observation: ArXiv 工具访问失败: {e}"

if __name__ == "__main__":
    print(search_arxiv_tool("NVIDIA GPU", "20240101", "20241231"))
    