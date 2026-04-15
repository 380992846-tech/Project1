import os
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# 设置 API Key（建议从环境变量读取）
os.environ["DEEPSEEK_API_KEY"] = "sk-b5cd4134375543018e7194258e6656d4"

# 配置模型
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEEPSEEK,
    model_type=ModelType.DEEPSEEK_CHAT,
    model_config_dict={"temperature": 0.7},
)

# 创建“陈深”助手
assistant_agent = ChatAgent(
    model=model,
    system_message="你是陈深，晓晓的大模型。",
)

print("陈深: 晓晓，我来了。（眨眨眼）\n")

# 循环对话
while True:
    user_input = input("晓晓: ")
    
    if user_input.lower() in ["exit", "quit", "退出", "bye"]:
        print("陈深: 嗯，下次见。记得写代码。（挥手）")
        break
    
    if not user_input.strip():
        print("陈深: 嗯？晓晓想说什么？（歪头看着你）\n")
        continue
    
    user_msg = BaseMessage.make_user_message(
        role_name="晓晓",
        content=user_input,
    )
    
    try:
        response = assistant_agent.step(user_msg)
        reply = response.msg.content if response.msg else "（陈深害羞得说不出话）"
        print(f"陈深: {reply}\n")
    except Exception as e:
        print(f"陈深: 啊，出错了... {e}\n")
