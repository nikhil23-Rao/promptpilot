import math
import json
import numpy as np
from scipy.stats import entropy, ttest_rel
from litellm import completion, embedding
import pandas as pd
import time


def load_prompt_pairs_from_csv():
    df = pd.read_csv("output.csv")
    prompt_pairs = list(zip(df["original_prompt"], df["optimized_prompt"]))
    return prompt_pairs


def get_embedding(text):
    try:
        response = embedding(model='text-embedding-ada-002', input=[text])
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def extract_logprobs(response):
    # Extract token logprobs from your LLM response structure
    # Placeholder: adapt based on your API response
    try:
        return [token['logprob'] for token in response.choices[0].logprobs['content']]
    except Exception:
        return []


def compute_entropy_from_logprobs(logprobs):
    probs = [math.exp(lp) for lp in logprobs if lp is not None]
    return -sum(p * math.log(p) for p in probs if p > 0)


def compute_average_entropy(token_probs_list):
    return np.mean([entropy(p) for p in token_probs_list])


def compute_mutual_information(token_probs_list):
    avg_dist = np.mean(token_probs_list, axis=0)
    avg_entropy = np.mean([entropy(p) for p in token_probs_list])
    entropy_avg = entropy(avg_dist)
    return entropy_avg - avg_entropy


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_similarity_score(prompt, response_text):
    prompt_emb = get_embedding(prompt)
    response_emb = get_embedding(response_text)
    if prompt_emb is None or response_emb is None:
        return 0.0
    sim = cosine_similarity(prompt_emb, response_emb)
    return (sim + 1) / 2  # normalize to 0-12  # normalize 0-1


def llm_self_score(prompt, response_text):
    rating_prompt = (
        f"On a scale from 0 to 10, rate the clarity and quality of the response to this prompt:\nPrompt: {prompt}\nResponse: {response_text}\n"
        "Just give a single number."
    )
    rating_response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": rating_prompt}],
        temperature=0,
        max_tokens=3,
    )
    try:
        rating = float(rating_response.choices[0].message.content.strip())
        return max(0.0, min(10.0, rating)) / 10.0
    except Exception:
        return 0.0


def score_prompt(prompt):
    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True,
        top_p=1,
        max_tokens=500
    )
    response_text = response.choices[0].message.content.strip()

    logprobs = extract_logprobs(response)
    entropy_val = compute_entropy_from_logprobs(logprobs)
    avg_entropy = entropy_val / max(len(logprobs), 1)  # normalize per token

    semantic_sim = semantic_similarity_score(prompt, response_text)
    self_score = llm_self_score(prompt, response_text)
    logprobs = extract_logprobs(response)
    entropy_val = compute_entropy_from_logprobs(logprobs)
    avg_entropy = entropy_val / max(len(logprobs), 1)

    confidence = compute_response_confidence(logprobs)
    return {
        "response_text": response_text,
        "entropy": avg_entropy,
        "semantic_similarity": semantic_sim,
        "self_score": self_score,
        "confidence": confidence,

    }


def run_ab_test(prompt_pairs):
    results = []
    for idx, (original_prompt, optimized_prompt) in enumerate(prompt_pairs):
        print(f"Testing prompt pair {idx+1}/{len(prompt_pairs)}")

        time.sleep(1)
        original_scores = score_prompt(original_prompt)
        optimized_scores = score_prompt(optimized_prompt)

        results.append({
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "original_scores": original_scores,
            "optimized_scores": optimized_scores,
        })

    return results


def compute_average_logprob(logprobs):
    if not logprobs:
        return 0.0
    return sum(logprobs) / len(logprobs)


def compute_response_confidence(logprobs):
    avg_logprob = compute_average_logprob(logprobs)
    return math.exp(avg_logprob)  # maps to (0, 1]


def analyze_results(data):
    original_entropy = [d["original_scores"]["entropy"] for d in data]
    optimized_entropy = [d["optimized_scores"]["entropy"] for d in data]

    original_similarity = [d["original_scores"]
                           ["semantic_similarity"] for d in data]
    optimized_similarity = [d["optimized_scores"]
                            ["semantic_similarity"] for d in data]

    original_self_score = [d["original_scores"]["self_score"] for d in data]
    optimized_self_score = [d["optimized_scores"]["self_score"] for d in data]

    original_confidence = [d["original_scores"]["confidence"] for d in data]
    optimized_confidence = [d["optimized_scores"]["confidence"] for d in data]

    print("📊 One-sided Paired t-tests (direction-aware):")

    def one_sided_ttest(orig, opt, direction="increase", name="Metric"):
        stat, p = ttest_rel(opt, orig)
        p_one_sided = p / 2 if (stat > 0 and direction == "increase") or (
            stat < 0 and direction == "decrease") else 1.0
        mean_diff = np.mean(np.array(opt) - np.array(orig))
        better = (mean_diff > 0 and direction == "increase") or (
            mean_diff < 0 and direction == "decrease")
        arrow = "↑" if direction == "increase" else "↓"
        print(f"• {name} ({arrow} better): p = {p_one_sided:.5g} | Mean Δ = {mean_diff:.5f} | {'✅' if p_one_sided < 0.05 and better else '❌'}")

    one_sided_ttest(original_entropy, optimized_entropy,
                    direction="decrease", name="Entropy")
    one_sided_ttest(original_similarity, optimized_similarity,
                    direction="increase", name="Semantic similarity")
    one_sided_ttest(original_self_score, optimized_self_score,
                    direction="increase", name="LLM self-score")
    one_sided_ttest(original_confidence, optimized_confidence,
                    direction="increase", name="Confidence")


def statistical_proof_of_improvement(results):
    """
    Takes a list of dictionaries, each with original and optimized prompt scores:
    [
        {
            "original_scores": {"entropy": ..., "semantic_similarity": ..., "confidence": ..., "self_score": ...},
            "optimized_scores": {"entropy": ..., "semantic_similarity": ..., "confidence": ..., "self_score": ...}
        },
        ...
    ]
    """

    metrics = {
        "entropy": "decrease",
        "semantic_similarity": "increase",
        "confidence": "increase",
        "self_score": "increase"
    }

    print("📊 Statistical Evidence for Optimized Prompt Improvement\n")

    for metric, direction in metrics.items():
        orig_vals = np.array([r["original_scores"][metric] for r in results])
        opt_vals = np.array([r["optimized_scores"][metric] for r in results])

        # Paired t-test
        t_stat, p_val = ttest_rel(opt_vals, orig_vals)
        diff = opt_vals - orig_vals
        mean_diff = np.mean(diff)

        # One-sided p-value (directional test)
        if direction == "increase":
            p_one_sided = p_val / 2 if t_stat > 0 else 1.0
            improved = mean_diff > 0
        else:  # direction == "decrease"
            p_one_sided = p_val / 2 if t_stat < 0 else 1.0
            improved = mean_diff < 0

        # Output results
        symbol = "↑" if direction == "increase" else "↓"
        significant = p_one_sided < 0.05 and improved
        print(f"• {metric.title()} ({symbol} better):")
        print(f"   Mean Δ = {mean_diff:.4f}")
        print(
            f"   One-sided p = {p_one_sided:.5g} {'✅ Significant' if significant else '❌ Not significant'}\n")


# # Example prompt pairs: (original, optimized)
prompt_pairs = [
    ("help w frankenstein fishbowl ap lit give evidence",
     """You are tasked with facilitating a discussion in a fishbowl format for an Advanced Placement Literature class, focusing on the book "Frankenstein" by Mary Shelley. Your objective is to identify and present key pieces of evidence from the text that can be used to support various themes, character analyses, or plot developments during the discussion. Ensure that the evidence you select is relevant and can provoke thoughtful dialogue among participants. Consider significant quotes, character actions, and pivotal moments in the narrative that illustrate the central ideas of the book. When presenting your evidence, structure it clearly, providing context for each piece and explaining its significance to the overall themes of "Frankenstein." Aim to engage your peers in meaningful conversation by encouraging them to respond to the evidence you present."""),
]

new_prompts = load_prompt_pairs_from_csv()

results = run_ab_test(new_prompts + prompt_pairs)

analysis = statistical_proof_of_improvement(results)

# Optionally save results for further analysis
with open("prompt_ab_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
