import os
from litellm import completion

response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a fun fact about dolphins."}]
)

print(response['choices'][0]['message']['content'])
