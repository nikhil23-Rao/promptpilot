from collections import Counter
import os
from litellm import completion
import math

rewritten_response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": """Write a polite and professional email to a colleague requesting a 30-minute meeting next week to discuss project updates. Use a formal tone and include a suggested time."""}],
    logprobs=True,
    temperature=0
)


original_response = completion(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": """Write an email to ask someone for a meeting."""}],
    logprobs=True,
    temperature=0
)


def get_new_prompt(original_prompt):
    system_message = (
        "You are an expert prompt engineer specializing in writing prompts that "
        "maximize LLM output confidence (i.e., higher average token probability). "
        "Rewrite the user's prompt so that the LLM's output will be more confident, "
        "clear, concise, and unambiguous."
    )

    user_message = (
        f"Original prompt:\n{original_prompt}\n\n"
        "Rewrite this prompt to maximize the LLM's confidence when answering. "
        "Make it clear, specific, and simple. Return the optimized prompt."
    )

    new_prompt = completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return new_prompt.choices[0].message.content


def compute_entropy_margin_score(response, alpha=0.5, beta=0.5):
    token_info_list = response['choices'][0]['logprobs']['content']

    entropies = []
    margins = []

    for token_info in token_info_list:
        top_logprobs = token_info.get("top_logprobs", [])
        if not top_logprobs or len(top_logprobs) < 2:
            continue

        # Entropy calculation
        probs = [math.exp(t['logprob']) for t in top_logprobs]
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs if total_prob > 0]

        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        entropies.append(entropy)

        # Logprob margin
        margin = top_logprobs[0]['logprob'] - top_logprobs[1]['logprob']
        margins.append(margin)

    if not entropies or not margins:
        return None

    avg_entropy = sum(entropies) / len(entropies)
    avg_margin = sum(margins) / len(margins)

    # Invert entropy to get confidence-like score
    entropy_conf = 1 / (1 + avg_entropy)

    # Final merged confidence score
    score = alpha * entropy_conf + beta * avg_margin

    return {
        "avg_entropy": avg_entropy,
        "avg_margin": avg_margin,
        "entropy_conf": entropy_conf,
        "confidence_score": score
    }


def compute_confidence_score_simple(response, alpha=0.5, beta=0.5):
    """
    Computes a confidence score as a percentage using entropy and logprob margin,
    without requiring predefined min/max values.
    """
    token_info_list = response['choices'][0]['logprobs']['content']
    entropies = []
    margins = []

    for token_info in token_info_list:
        top_logprobs = token_info.get("top_logprobs", [])
        if not top_logprobs or len(top_logprobs) < 2:
            continue

        # --- Entropy ---
        probs = [math.exp(t['logprob']) for t in top_logprobs]
        total_prob = sum(probs)
        if total_prob == 0:
            continue
        probs = [p / total_prob for p in probs]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        entropies.append(entropy)

        # --- Margin ---
        margin = top_logprobs[0]['logprob'] - top_logprobs[1]['logprob']
        margins.append(margin)

    if not entropies or not margins:
        return None

    avg_entropy = sum(entropies) / len(entropies)
    avg_margin = sum(margins) / len(margins)

    # Bounded [0, 1] components
    entropy_conf = 1 / (1 + avg_entropy)
    margin_conf = avg_margin / (1 + avg_margin)

    # Weighted combined confidence (still [0, 1])
    raw_score = alpha * entropy_conf + beta * margin_conf
    percent_score = round(raw_score * 100, 2)

    return {
        "avg_entropy": avg_entropy,
        "avg_margin": avg_margin,
        "entropy_conf": entropy_conf,
        "margin_conf": margin_conf,
        "confidence_score": raw_score,
        "confidence_percent": percent_score
    }


def compute_full_confidence_score(response,
                                  alpha=0.3,   # entropy confidence weight
                                  beta=0.3,    # margin confidence weight
                                  gamma=0.2,   # top-1 prob weight
                                  delta=0.1,   # length bonus weight
                                  epsilon=0.1,  # repetition penalty weight
                                  ideal_length=50):
    """
    Computes a full composite confidence score from an LLM response.
    Returns both raw score and percentage [0-100].
    """

    token_info_list = response['choices'][0]['logprobs']['content']
    if not token_info_list:
        return None

    entropies = []
    margins = []
    top1_probs = []
    token_texts = []

    for token_info in token_info_list:
        top_logprobs = token_info.get("top_logprobs", [])
        if not top_logprobs or len(top_logprobs) < 2:
            continue

        # Token text
        token_texts.append(token_info.get('token', ''))

        # Entropy
        probs = [math.exp(t['logprob']) for t in top_logprobs]
        total_prob = sum(probs)
        if total_prob == 0:
            continue
        probs = [p / total_prob for p in probs]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        entropies.append(entropy)

        # Logprob margin
        margin = top_logprobs[0]['logprob'] - top_logprobs[1]['logprob']
        margins.append(margin)

        # Top1 prob
        top1_probs.append(math.exp(top_logprobs[0]['logprob']))

    if not entropies or not margins or not top1_probs:
        return None

    # --- Calculate individual components ---
    avg_entropy = sum(entropies) / len(entropies)
    entropy_conf = 1 / (1 + avg_entropy)

    avg_margin = sum(margins) / len(margins)
    margin_conf = avg_margin / (1 + avg_margin)

    avg_top1_prob = sum(top1_probs) / len(top1_probs)

    # Length bonus
    length = len(token_texts)
    length_score = min(1.0, length / ideal_length)

    # Repetition penalty
    token_counts = Counter(token_texts)
    repeated = [t for t, c in token_counts.items() if c > 1]
    rep_rate = sum(token_counts[t]
                   for t in repeated) / length if length > 0 else 0
    repetition_penalty = 1.0 - rep_rate

    # --- Final weighted score ---
    raw_score = (
        alpha * entropy_conf +
        beta * margin_conf +
        gamma * avg_top1_prob +
        delta * length_score +
        epsilon * repetition_penalty
    )

    # Normalize to 0–100%
    percent_score = round(raw_score * 100, 2)

    return {
        "avg_entropy": avg_entropy,
        "entropy_conf": entropy_conf,
        "avg_margin": avg_margin,
        "margin_conf": margin_conf,
        "avg_top1_prob": avg_top1_prob,
        "length_score": length_score,
        "repetition_penalty": repetition_penalty,
        "confidence_score": raw_score,
        "confidence_percent": percent_score
    }


def get_average_logprob_from_modelresponse(response):
    """
    Extract log probabilities from a LiteLLM ModelResponse object and compute average log prob.

    Args:
        response (ModelResponse): The object returned by LiteLLM's completion call.

    Returns:
        float: Average log probability (confidence) of the generated tokens.
    """
    # Step 1: Navigate to token logprobs
    try:
        token_logprobs = [
            token_data["logprob"]
            for token_data in response.choices[0].logprobs['content']
            if "logprob" in token_data
        ]
    except Exception as e:
        print(f"Error extracting logprobs: {e}")
        return float('-inf')

    # Step 2: Return average logprob
    if not token_logprobs:
        return float('-inf')

    return sum(token_logprobs) / len(token_logprobs)


# print(get_new_prompt("photosynthesis -> make easy"))
avgOld = get_average_logprob_from_modelresponse(original_response)
avgNew = get_average_logprob_from_modelresponse(rewritten_response)

print("OLD ", math.exp(avgOld))
print("NEW ", math.exp(avgNew))
