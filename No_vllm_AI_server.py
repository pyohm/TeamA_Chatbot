from openai import OpenAI

  client = OpenAI(api_key= "여기에 API_KEY를 넣으세요",
                base_url= "https://api.groq.com/openai/v1")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.2-90b-vision-preview",
    stream=False,
)

print(chat_completion.choices[0].message.content)
