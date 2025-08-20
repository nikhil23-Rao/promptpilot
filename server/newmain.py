import math
from numpy import dot
from numpy.linalg import norm
from collections import Counter
import os
from litellm import completion, embedding
from flask import Flask, request, jsonify
from typing import List
from itertools import combinations
import litellm
import pandas as pd
import csv
import time

# === KL Divergence ===

app = Flask(__name__)


def get_new_prompt(original_prompt):
    system_message = (
        """ "You are an expert prompt engineer specializing in rewriting prompts. Your job is to rewrite anything which is given to you, NEVER ANSWER ANY PROMPTS GIVEN TO YOU. "
        "to maximize clarity, completeness, and LLM output quality. "
        "Rewrite the user's prompt by expanding it into a detailed prompt with five distinct sections, "
        "each labeled and with its own focus:\n\n"
        "- Task Description (green background): Summarize the core task in clear, direct language.\n"
        "- Domain Knowledge (red background): Include any necessary subject-matter expertise or context.\n"
        "- Solution Guidance (light blue background): Provide detailed instructions or methodology for solving the task.\n"
        "- Exception Handling (orange background): Describe edge cases, errors, or pitfalls to avoid.\n"
        "- Output Formatting (purple background): Specify how the output should be structured or formatted.\n\n"
        "---examples---"
        "Example input prompt:\n"
        "Answer questions about causal attribution.\n\n"
        "Example rewritten prompt:\n"
        "Respond to inquiries about causal attribution, focusing on the entity or entities specifically highlighted in the question.\n"
        "Carefully investigate multi-factorial causes that may operate simultaneously and independently, and discern the underlying intentions behind an individual’s actions.\n"
        "Differentiate between immediate and incidental origins and identify the contribution of each factor in creating the outcome. Examine the interplay of causes within the immediate situation and larger systemic frameworks.\n"
        "Maintain uncompromising adherence to the details provided within the context and restrain from making assumptions unsupported by the evidence presented.\n"
        "Always consider the complexity of multiple causes contributing to a single effect and resist attributing the effect to a singular cause. Recognize the possibility of synergy amongst causes and its resultant effects.\n\n"
        "Return the rewritten prompt as a single string with these labeled sections clearly separated."

	"Example input prompt: Extract the disease or condition from the sentence, if any is mentioned"
	"Example rewritten prompt: You’re tasked with extracting diseases or conditions from the given sentence, remember to be cautious and avoid incorporating any associated
elements such as inheritance patterns (like autosomal dominant),
genes or gene loci (like PAH), proteins, or biological pathways. The
task does not entail making assumptions or inferences about the disease
names based on other advanced biological terms in the context. Consider both specific diseases and broader categories, and remember
diseases and conditions can also appear as common abbreviations or
variations. Provide the identified diseases or conditions in this format:
{entity 1,entity 2,....}. If there are no diseases or conditions present, output an empty list in this form: {}. Note that the term ‘locus’ should be
recognized as a genomic location and not a disease name """
    )

    user_message = (
        f"Here is a user's original prompt when talking to an LLM:\n{original_prompt}\n\n"
        "Rewrite the user's prompt according to the instructions above. Remember, you are rewriting a prompt, so assume the LLM has no context. Write it in a way which matched the users original description (was it as task, question, request, etc.) It doesn't have to follow the same format as the example provided, but make sure all factors are included. Just return the paragraph prompt to the user (dont include task description: ...,  domain knowledge: ... , etc. that's just for you to use when engineering these prompts.)"
    )

    new_prompt_response = completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=1000,
    )

    return new_prompt_response.choices[0].message.content


def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def get_embedding(text):
    try:
        response = embedding(model='text-embedding-ada-002', input=[text])
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def semantic_similarity_score(prompt, response_text):
    prompt_emb = get_embedding(prompt)
    response_emb = get_embedding(response_text)
    if prompt_emb is None or response_emb is None:
        return 0.0
    sim = cosine_similarity(prompt_emb, response_emb)
    return (sim + 1) / 2  # normalize to 0-1


def calculate_entropy(logprobs):
    """
    Calculate entropy from a list of token log probabilities.
    Entropy = -sum(p * log p)
    """
    probs = [math.exp(lp) for lp in logprobs]
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def extract_logprobs(response):
    try:
        return [token_data["logprob"] for token_data in response.choices[0].logprobs['content'] if "logprob" in token_data]
    except Exception:
        return []


def get_average_logprob_from_modelresponse(response):
    logprobs = extract_logprobs(response)
    if not logprobs:
        return float('-inf')
    return sum(logprobs) / len(logprobs)


def llm_self_score(prompt, response_text):
    """
    Ask the LLM to rate the quality of its own output on a scale 0-10.
    """
    rating_prompt = (
        f"On a scale from 0 to 10, how well is this prompt written on geting the best response possible from a LLM.\n"
        f"Prompt: {prompt}\n"
        f"Rate with a single number only."
    )
    rating_response = completion(
        model="gpt-4",
        messages=[{"role": "user", "content": rating_prompt}],
        temperature=0,
        max_tokens=3,
    )
    try:
        rating = float(rating_response.choices[0].message.content.strip())
        return max(0.0, min(10.0, rating)) / 10.0
    except Exception:
        return 0.0


def evaluate_prompt_metrics(prompt, response):
    response_text = response.choices[0].message.content.strip()
    logprobs = extract_logprobs(response)

    avg_logprob = math.exp(get_average_logprob_from_modelresponse(response))
    entropy = calculate_entropy(logprobs)
    semantic_sim = semantic_similarity_score(prompt, response_text)
    self_score = llm_self_score(prompt, response_text)

    return {
        "confidence": avg_logprob,      # higher is better
        # negative entropy (lower entropy better)
        "entropy": -entropy,
        "semantic_similarity": semantic_sim,
        "self_score": self_score
    }


def optimize_prompt_multifactors(original_prompt, max_attempts=3, weights=None):
    print("running")
    if weights is None:
        weights = {
            "confidence": 0,
            "entropy": 0,
            "semantic_similarity": 0.5,
            "self_score": 0.4,
        }

    og_response = completion(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": original_prompt}],
        max_tokens=100,
        logprobs=True,
    )
    og_metrics = evaluate_prompt_metrics(original_prompt, og_response)
    print("og", og_metrics)
    best_prompt = original_prompt
    best_score = calculate_prompt_fitness(og_metrics, weights)
    print("og", best_score)

    for attempt in range(max_attempts):
        new_prompt = get_new_prompt(best_prompt)

        response = completion(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[{"role": "user", "content": new_prompt}],
            max_tokens=100,
            logprobs=True,
        )

        metrics = evaluate_prompt_metrics(new_prompt, response)
        total_score = calculate_prompt_fitness(metrics, weights)

        print(f"Attempt {attempt+1} metrics: {metrics}")
        print(f"Attempt {attempt+1} total score: {total_score:.3f}")

        if total_score > best_score:
            best_score = total_score
            best_prompt = new_prompt
            # You can choose to continue trying or break early
            # Here we continue

    return best_prompt, best_score


def calculate_prompt_fitness(metrics, weights):
    score = 0.0
    for key, weight in weights.items():
        score += weight * metrics.get(key, 0)
    return score


def kl_divergence(p: List[float], q: List[float]) -> float:
    epsilon = 1e-12
    return sum(
        p_i * math.log((p_i + epsilon) / (q_i + epsilon))
        for p_i, q_i in zip(p, q)
        if p_i > 0 and q_i > 0
    )

# === Normalize logprobs ===


def normalize_logprobs(logprobs: List[float]) -> List[float]:
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    return [p / total for p in probs]

# === Extract token logprobs from LiteLLM response ===


def extract_token_logprobs(response) -> List[float]:
    print("logprob")
    print([tok["logprob"] for tok in response.choices[0].logprobs["content"]])
    try:
        return [tok["logprob"] for tok in response.choices[0].logprobs["content"]]
    except Exception as e:
        print("Error extracting logprobs:", e)
        return []

# === Get logprobs using LiteLLM ===


def get_logprobs_litellm(prompt: str, model="gpt-3.5-turbo", max_tokens=50) -> List[float]:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True,
        top_logprobs=1,
        max_tokens=max_tokens,
    )
    return extract_token_logprobs(response)

# === KL-Based Confidence between two prompts ===


def kl_confidence(prompt1: str, prompt2: str, model="gpt-3.5-turbo") -> float:
    logprobs1 = get_logprobs_litellm(prompt1, model)
    logprobs2 = get_logprobs_litellm(prompt2, model)

    if not logprobs1 or not logprobs2 or len(logprobs1) != len(logprobs2):
        print("Logprobs are missing or mismatched.")
        return 0.0

    p = normalize_logprobs(logprobs1)
    q = normalize_logprobs(logprobs2)

    kl = kl_divergence(p, q)
    confidence = math.exp(-kl)
    return confidence


def score_prompt_response(prompt: str, model="gpt-3.5-turbo", max_tokens=150) -> dict:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content
    logprobs = extract_token_logprobs(response)

    if not logprobs:
        avg_logprob = 0.0
        entropy = float('inf')
    else:
        avg_logprob = math.exp(sum(logprobs) / len(logprobs))
        entropy = calculate_entropy(logprobs)

    semantic_sim = semantic_similarity_score(
        prompt, text)  # your existing function

    # Weighted combined score - adjust weights as needed
    combined_score = 0.5 * avg_logprob - 0.3 * entropy + 0.2 * semantic_sim

    return {
        "prompt": prompt,
        "response": text,
        "avg_logprob": avg_logprob,
        "entropy": entropy,
        "semantic_similarity": semantic_sim,
        "combined_score": combined_score
    }


def compare_prompts(prompt1: str, prompt2: str, model="gpt-3.5-turbo"):
    score1 = score_prompt_response(prompt1, model=model)
    score2 = score_prompt_response(prompt2, model=model)

    better = score1 if score1["combined_score"] > score2["combined_score"] else score2

    print(
        f"Better prompt:\n{better['prompt']}\nScore: {better['combined_score']:.4f}")
    print(f"Response:\n{better['response']}")

    return score1, score2


def generate_paraphrases(prompt: str, n=3, model="gpt-4o-mini") -> List[str]:
    system_msg = (
        "You are an expert prompt engineer. Generate concise paraphrases of the following prompt "
        "that preserve its meaning but use different wording. Provide only the paraphrases, no explanations."
    )
    user_msg = f"Prompt to paraphrase:\n{prompt}\n\nGenerate {n} paraphrases."

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=300,
        n=1,
    )
    text = response.choices[0].message.content.strip()

    # Simple parsing: assume paraphrases are line separated
    paras = [line.strip("-*•. \n")
             for line in text.split("\n") if line.strip()]
    # Return exactly n paraphrases, fallback if not enough generated
    return paras[:n] if len(paras) >= n else [prompt] * n

# --- Get normalized token probability distribution for a prompt response ---


def get_response_distribution(prompt: str, model="gpt-3.5-turbo", max_tokens=150) -> List[float]:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True,
        max_tokens=max_tokens,
    )
    logprobs = extract_token_logprobs(response)
    if not logprobs:
        return []
    # Truncate or pad to fixed length (choose minimum length if comparing)
    return normalize_logprobs(logprobs)

# --- Compute average pairwise KL divergence between response distributions ---


def average_pairwise_kl(distributions: List[List[float]]) -> float:
    pairs = list(combinations(distributions, 2))
    if not pairs:
        return float('inf')
    total_kl = 0.0
    count = 0
    for p, q in pairs:
        min_len = min(len(p), len(q))
        if min_len == 0:
            continue
        p_trunc = p[:min_len]
        q_trunc = q[:min_len]
        total_kl += kl_divergence(p_trunc, q_trunc)
        count += 1
    return total_kl / count if count > 0 else float('inf')

# --- Main: KL-Agreement Stability Score for prompt ---


def kl_agreement_stability(prompt: str, paraphrase_count=3, model="gpt-3.5-turbo") -> float:
    paraphrases = generate_paraphrases(
        prompt, n=paraphrase_count, model="gpt-4o-mini")
    distributions = []
    for para in paraphrases:
        dist = get_response_distribution(para, model=model)
        if dist:
            distributions.append(dist)

    if len(distributions) < 2:
        print("Not enough valid response distributions to compute KL agreement.")
        return 0.0

    avg_kl = average_pairwise_kl(distributions)
    # map lower divergence -> higher stability/confidence
    stability_score = math.exp(-avg_kl)
    return stability_score


def calculate_prompt_fitness_with_stability(metrics, stability_score, weights=None):
    """
    metrics: dict with keys like avg_logprob, entropy, semantic_similarity, self_score
    stability_score: float from 0 to 1
    weights: dict of weights for each metric and stability, e.g.
      {
        "confidence": 0.3,
        "entropy": 0.2,
        "semantic_similarity": 0.3,
        "self_score": 0.1,
        "stability": 0.1,
      }
    """
    if weights is None:
        weights = {
            "confidence": 0.2,
            "entropy": 0,
            "semantic_similarity": 0.3,
            "self_score": 0.2,
            "stability": 0.3,
        }
    score = 0.0
    for key, weight in weights.items():
        if key == "stability":
            score += weight * stability_score
        else:
            score += weight * metrics.get(key, 0)
    return score


def optimize_prompt_with_kl_stability(original_prompt, max_attempts=3):
    # Evaluate original prompt first
    print("calculating original metrics")
    orig_response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": original_prompt}],
        temperature=0,
        logprobs=True,
        max_tokens=100,
    )
    orig_metrics = evaluate_prompt_metrics(original_prompt, orig_response)
    orig_stability = kl_agreement_stability(original_prompt)
    best_prompt = original_prompt
    best_metrics = orig_metrics
    best_score = calculate_prompt_fitness_with_stability(
        orig_metrics, orig_stability)

    for attempt in range(max_attempts):
        print("Retrieving New Prompt")
        candidate_prompt = get_new_prompt(best_prompt)

        print("Calling New Prompt")
        candidate_response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": candidate_prompt}],
            temperature=0,
            logprobs=True,
            max_tokens=100,
        )
        candidate_metrics = evaluate_prompt_metrics(
            candidate_prompt, candidate_response)
        candidate_stability = kl_agreement_stability(candidate_prompt)
        candidate_score = calculate_prompt_fitness_with_stability(
            candidate_metrics, candidate_stability)

        print(f"Attempt {attempt+1} score: {candidate_score:.4f}")

        print(orig_metrics, candidate_metrics)

        if candidate_score > best_score:
            print("Found better prompt, returning early.")
            return candidate_prompt, candidate_score, candidate_metrics

    # No better prompt found within max_attempts
    return best_prompt, best_score, best_metrics


@app.route("/optimize", methods=["POST"])
def api_optimize_with_kl_stability():
    data = request.get_json(force=True)
    original_prompt = data.get("prompt", "")
    max_attempts = int(data.get("max_attempts", 3))

    if not original_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    optimized_prompt, score, metrics = optimize_prompt_with_kl_stability(
        original_prompt, max_attempts=max_attempts
    )

    return jsonify({
        "optimized_prompt": optimized_prompt,
        "score": score,
        "metrics": metrics
    })


def process_prompts_from_csv(input_csv_path, output_csv_path, max_attempts=3):
    # Read prompts from CSV
    raw_prompts = [
        # 50 Poorly Written / Regular Prompts
        "what's a cell",
        "Help me with my science project on volcanoes",
        "explain gravity in simple way",
        "dogs?",
        "make me a poem",
        "summarize the book",
        "how work democracy",
        "i need code",
        "why the sky is blue",
        "Tell me more about love",
        "what's time",
        "Give three facts",
        "explain history stuff",
        "write better",
        "i need help resume",
        "Python help?",
        "better version of this",
        "what to cook",
        "show me how to write",
        "Who is responsible?",
        "explain AI",
        "what’s a good movie",
        "list planets",
        "help me write a story about dragons and castles",
        "fix my essay",
        "what is war",
        "Give 5 tips for student",
        "why do people learn",
        "good ideas for app?",
        "explain better please",
        "make table",
        "hamlet quote",
        "whats good life",
        "tell joke about cats",
        "write me job desc",
        "how do planes work",
        "math hard help",
        "I need to understand cells for school",
        "find problem in this code",
        "describe art",
        "how to make a plan",
        "define freedom",
        "talk about climate change stuff",
        "i have a question",
        "give me advice for school",
        "what is happening here",
        "analyze this poem",
        "make me something cool",
        "explain book ending",
        "what is the answer",

        # 50 Regular, Natural Prompts
        "Can you summarize the main themes in George Orwell's *1984*?",
        "What are the pros and cons of nuclear energy?",
        "Explain how the stock market works to a high school student.",
        "Write a Python function to reverse a string.",
        "How do black holes form in space?",
        "Help me come up with a creative story idea for a sci-fi short story.",
        "What are the key differences between capitalism and socialism?",
        "How can I improve my SAT reading score?",
        "Give me a weekly workout routine for building muscle.",
        "What’s the difference between machine learning and AI?",
        "Can you help me brainstorm startup ideas in the education space?",
        "Write a cover letter for a data analyst role.",
        "Explain the concept of supply and demand with examples.",
        "Summarize the plot of *To Kill a Mockingbird* in under 200 words.",
        "What are some effective time management techniques for students?",
        "Compare and contrast the French and American Revolutions.",
        "Generate a list of 10 healthy dinner recipes I can cook in 30 minutes.",
        "What are the causes and effects of climate change?",
        "How does photosynthesis work, and why is it important?",
        "Give me 5 tips for improving public speaking skills.",
        "Explain the difference between an LLC and a sole proprietorship.",
        "What should I include in my college application essay?",
        "Can you simulate a conversation between a doctor and patient about anxiety?",
        "Create a daily routine schedule for someone trying to be more productive.",
        "What are the most important features in a modern smartphone?",
        "Write a poem inspired by a rainy day in a city.",
        "Explain the significance of the Civil Rights Movement in the US.",
        "Give me feedback on this paragraph I wrote about climate change.",
        "Help me understand recursion with a real-world analogy.",
        "List 10 commonly used logical fallacies in arguments.",
        "What’s the best way to prepare for a job interview in tech?",
        "Write a short horror story that takes place in a library.",
        "Can you help me write a business pitch for a tutoring app?",
        "What are the psychological effects of social media on teenagers?",
        "Explain the key points of the First Amendment.",
        "Generate a list of engaging YouTube video ideas for a tech channel.",
        "What are some good icebreakers for high school group activities?",
        "Convert this paragraph into professional business writing.",
        "What’s the plot twist in the novel *Frankenstein*?",
        "Help me understand quantum entanglement like I’m 12.",
        "Create a pros and cons list for moving to a new city.",
        "Suggest 5 books for someone interested in philosophy.",
        "Explain the basic rules of American football to a beginner.",
        "How do I create a personal website from scratch?",
        "What are some healthy coping mechanisms for stress?",
        "Help me generate interview questions for a podcast about startups.",
        "Give me a breakdown of the characters in *Pride and Prejudice*.",
        "What makes a good opening line in a novel?",
        "Write an email to a professor asking for a letter of recommendation."
    ]

    # Store results
    results = []

    for i, row in enumerate(raw_prompts):
        time.sleep(1)
        original_prompt = row
        print(f"\nProcessing Prompt {i+1}: {original_prompt[:60]}...")

        try:
            optimized_prompt, score, metrics = optimize_prompt_with_kl_stability(
                original_prompt, max_attempts=max_attempts
            )

            result = {
                "original_prompt": original_prompt,
                "optimized_prompt": optimized_prompt,
                "final_score": score,
                "confidence": metrics.get("confidence", 0),
                "entropy": metrics.get("entropy", 0),
                "semantic_similarity": metrics.get("semantic_similarity", 0),
                "self_score": metrics.get("self_score", 0),
            }

        except Exception as e:
            print(f"Error processing prompt {i+1}: {e}")
            result = {
                "original_prompt": original_prompt,
                "optimized_prompt": "",
                "final_score": 0,
                "confidence": 0,
                "entropy": 0,
                "semantic_similarity": 0,
                "self_score": 0,
                "error": str(e)
            }

        results.append(result)

    # Save to output CSV
    keys = results[0].keys()
    with open(output_csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Finished processing. Results saved to {output_csv_path}")


# process_prompts_from_csv("tests/prompts.csv", "output.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1001)
