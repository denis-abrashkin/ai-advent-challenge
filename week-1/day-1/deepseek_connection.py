import os
from openai import OpenAI

api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
if not api_key:
    raise RuntimeError("ANTHROPIC_AUTH_TOKEN не найден — проверьте экспорт переменной")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

print("DeepSeek CLI (введите 'exit' для выхода)\n")

while True:
    user_input = input("Вы: ").strip()
    if user_input.lower() in ("exit", "quit", "выход"):
        break
    if not user_input:
        continue

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Do not use emojis in your responses."},
            {"role": "user", "content": user_input},
        ],
    )

    print(f"\nDeepSeek: {response.choices[0].message.content}\n")
