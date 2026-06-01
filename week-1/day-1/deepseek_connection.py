import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY не найден — проверьте .env или экспорт переменной")

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
