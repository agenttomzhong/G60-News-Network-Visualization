import pandas as pd
import json
import time
from openai import OpenAI
from tqdm import tqdm
import os
import datetime

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo-latest"
INPUT_CSV = r"D:\Documents\Python\Proj25.06.04\WorkData\frequency\sum.csv"
RETRY_LIMIT = 5 

os.makedirs("EntityExtraction/EntityDict/result", exist_ok=True)
OUTPUT_PAIRS_CSV = "EntityExtraction/EntityDict/result/entity_dict.csv"

def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head(10))
    required_columns = ['word', 'frequency']
    # Ensure required columns exist
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns {required_columns}")
    # Sort by frequency descending
    df = df.sort_values(ascending=False, by='frequency')
    df = df[required_columns]
    df = df.head(300)
    print(df.tail(10))
    print(f"Loaded {len(df)} rows")
    return df[required_columns]

def build_system_prompt():
    system_prompt = """
    你是一个中文文本分析助手，需要根据提供的词频表筛选出60个核心实体，分为以下三类：  
- **产业类**（20个）  
- **技术类**（20个）  
- **资本类**（20个）  

请按照以下规则进行分类与筛选。
---

### 1. **输入格式**

提供一份词频表，每行为一个条目，格式为 `词语: 频率`，例如：
集成电路: 150  
专利: 120  
基金: 95  
---

### 2. **筛选规则**

- **排除虚词和通用行政词汇**（如“的”、“了”、“召开”、“发展”等）。
- **优先高频词**：频率越高优先级越高（但需结合语义）。
- **按类别分类**：
  - **产业类**：具体行业或领域（如集成电路、新能源汽车）
  - **技术类**：技术术语或创新方法（如专利、工业互联网）
  - **资本类**：金融工具或资金相关词（如基金、科创板）
- 若发现潜在新领域词汇（如“氢能”），即使频率较低也可标注建议性理由。
- 不允许添加在输入的词频表中未出现的实体词。

---

### 3. **输出要求**

严格按以下格式输出 JSON：

```json
{
  "result": [
    {
      "category": "产业/技术/资本",
      "entity": "实体词",
      "frequency": 150,
      "reason": "筛选理由"
    }
  ]
}
```

字段说明：
- `category`: 实体所属类别（产业/技术/资本）
- `entity`: 实体词名称
- `frequency`: 原始词频
- `reason`: 简要说明筛选理由（不超过30字）

---

### 4. **增强验证机制**

- 自动检测重复实体或语义不匹配项，标记时给出清晰理由。
- 不允许添加在输入的词频表中未出现的实体词。
- 不允许输出与类型名称相同的实体词，例如不能输出“产业”作为实体词。
---

### 5. **示例**

#### 输入示例：
集成电路: 150  
专利: 120  
基金: 95  


#### 输出示例：
```json
{
  "result": [
    {"category": "产业", "entity": "集成电路", "frequency": 150, "reason": "国家重点发展的战略性新兴产业"},
    {"category": "技术", "entity": "专利", "frequency": 120, "reason": "衡量技术创新的重要指标"},
    {"category": "资本", "entity": "基金", "frequency": 95, "reason": "支持科技创新的主要金融工具"},
  ]
}
直接输出纯JSON，禁止任何解释文本。
"""
    return system_prompt

def call_generation_api(client, system_prompt, input_data):
    # Call API with retry mechanism
    for attempt in range(RETRY_LIMIT):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)}
                ],
                timeout=30
            )
            response = completion.choices[0].message.content.strip()
            return json.loads(response)
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{RETRY_LIMIT}): {str(e)}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
    return None

def prepare_input_data(df):
    # Prepare input data for API
    return {"data": df.to_dict(orient="records")}

def main():
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    df = load_csv_data(INPUT_CSV)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows")

    # Build system prompt and input data
    system_prompt = build_system_prompt()
    input_data = prepare_input_data(df)

    print("Calling API to generate entity classification result...")
    result = call_generation_api(client, system_prompt, input_data)
    if result is None:
        print("API call failed, no result obtained.")
        return

    # Save result to CSV
    result_df = pd.DataFrame(result['result'])
    result_df.to_csv(OUTPUT_PAIRS_CSV, index=False, encoding='utf-8-sig')
    print(f"Result saved to {OUTPUT_PAIRS_CSV}")

if __name__ == "__main__":
    main()