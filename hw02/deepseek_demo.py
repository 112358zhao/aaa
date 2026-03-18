# deepseek_demo1.py
import os
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="9cbb35ad-5c8d-440c-b0b8-d46a86da10be", # 填写你的API Key
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 设置问题
question = "常见的十字花科植物有哪些？"

# 发送请求
response = client.chat.completions.create(
    model="ep-20260318222559-ld2dq",  # 替换为你的"模型接入点 ID"
    messages=[
        {"role":"user","content": question}
    ]
)

# DeepSeek模型特有功能：输出推理过程
# 当模型触发深度推理时，会生成reasoning_content
if hasattr(response.choices[0].message,'reasoning_content'):
# 打印模型的思维链/推理过程
    print(response.choices[0].message.reasoning_content)

# 打印最终答案
print(response.choices[0].message.content)