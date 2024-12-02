from openai import OpenAI

client = OpenAI(api_key= "EMPTY",
                base_url= "http://axonflow.xyz/v1")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "안녕",
        }
    ],
    model="Qwen/Qwen2-VL-2B-Instruct",
    stream=False,
)

print(chat_completion.choices[0].message.content)
