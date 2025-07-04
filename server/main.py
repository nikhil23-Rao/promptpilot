import os
from litellm import completion
import math

response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write CSS code to create a smooth, animated wave effect that can be used as a decorative background. It should loop infinitely and be responsive across screen sizes."}],
    logprobs=True,
    top_logprobs=4
)

# extract token logprobs from your response JSON structure
token_info_list = response['choices'][0]['logprobs']['content']

print(token_info_list)

# Get the logprob for each token (filter out None just in case)
token_logprobs = [t['logprob']
                  for t in token_info_list if t['logprob'] is not None]


token_confidence_margins = []
for token_info in token_info_list:
    top_probs = token_info.get('top_logprobs', [])
    if len(top_probs) >= 2:
        # logit margin between top-1 and top-2
        margin = top_probs[0]['logprob'] - top_probs[1]['logprob']
        token_confidence_margins.append(margin)

avg_margin = sum(token_confidence_margins) / len(token_confidence_margins)
print(f"Average logprob margin: {avg_margin:.4f}")

# # Sum log probs (log prob of entire sequence)
# total_logprob = sum(token_logprobs)

# # Convert logprob to probability (confidence score)
# response_confidence = math.exp(total_logprob)
# num_tokens = len(token_logprobs)
# avg_logprob = total_logprob / num_tokens
# avg_prob = math.exp(avg_logprob)

# print(f"Number of tokens: {num_tokens}")
# print(f"Average token logprob: {avg_logprob}")
# print(f"Average token probability: {avg_prob}")

# print(
#     f"Confidence score of entire response (joint probability): {response_confidence:e}")
