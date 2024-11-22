from openai import OpenAI

client = OpenAI(
    base_url="http://axonflow.xyz/v1",
    api_key="eodbs7149",
)

completion = client.chat.completions.create(
  model="meta-llama/Llama-3.2-3B-Instruct",
  max_tokens=300,
  messages=[
    {"role": "user", "content": "안녕?"}
  ]
)

print(completion.choices[0].message)