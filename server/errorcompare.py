import math
import json
import numpy as np
from scipy.stats import entropy, ttest_rel
from litellm import completion, embedding


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


def analyze_results(results):
    entropy_diffs = []
    semantic_diffs = []
    self_score_diffs = []
    conf_diffs = []

    for r in results:
        oe = r['original_scores']['entropy']
        osim = r['original_scores']['semantic_similarity']
        oscore = r['original_scores']['self_score']
        oc = r['original_scores']['confidence']

        pe = r['optimized_scores']['entropy']
        psim = r['optimized_scores']['semantic_similarity']
        pscore = r['optimized_scores']['self_score']
        pc = r['optimized_scores']['confidence']

        # Replace None or invalid values with NaN
        values = [oe, osim, oscore, oc, pe, psim, pscore, pc]
        values = [float('nan') if v is None else v for v in values]

        if not any(map(math.isnan, values)):
            # Lower entropy is better
            entropy_diffs.append(oe - pe)
            # Higher similarity is better
            semantic_diffs.append(psim - osim)
            # Higher self-score is better
            self_score_diffs.append(pscore - oscore)
            # Higher confidence is better
            conf_diffs.append(pc - oc)
        else:
            print(
                f"Skipping due to NaN in scores: {r['original_prompt'][:60]}...")

    if len(entropy_diffs) == 0:
        print("No valid data to analyze.")
        return {}

    print("\n🔍 Paired t-tests (optimized vs original):")
    print("• Entropy (↓ better):          p =", ttest_rel(
        entropy_diffs, np.zeros(len(entropy_diffs))).pvalue)
    print("• Semantic similarity (↑):     p =", ttest_rel(
        semantic_diffs, np.zeros(len(semantic_diffs))).pvalue)
    print("• LLM self-score (↑):          p =",
          ttest_rel(self_score_diffs, np.zeros(len(self_score_diffs))).pvalue)
    print("• Confidence (↑):              p =", ttest_rel(
        conf_diffs, np.zeros(len(conf_diffs))).pvalue)

    win_rate = sum(
        (ed > 0 and sd > 0 and sc > 0 and cd > 0)
        for ed, sd, sc, cd in zip(entropy_diffs, semantic_diffs, self_score_diffs, conf_diffs)
    ) / len(entropy_diffs)

    print(f"\n🏆 Win rate (all 4 metrics improved): {win_rate*100:.1f}%")

    return {
        "entropy_diffs": entropy_diffs,
        "semantic_diffs": semantic_diffs,
        "self_score_diffs": self_score_diffs,
        "confidence_diffs": conf_diffs,
        "win_rate": win_rate,
    }


# Example prompt pairs: (original, optimized)
prompt_pairs = [
    ("help w frankenstein fishbowl ap lit give evidence",
     """You are tasked with facilitating a discussion in a fishbowl format for an Advanced Placement Literature class, focusing on the book "Frankenstein" by Mary Shelley. Your objective is to identify and present key pieces of evidence from the text that can be used to support various themes, character analyses, or plot developments during the discussion. Ensure that the evidence you select is relevant and can provoke thoughtful dialogue among participants. Consider significant quotes, character actions, and pivotal moments in the narrative that illustrate the central ideas of the book. When presenting your evidence, structure it clearly, providing context for each piece and explaining its significance to the overall themes of "Frankenstein." Aim to engage your peers in meaningful conversation by encouraging them to respond to the evidence you present."""),
    ("code css wave hero page react",
        "Your task is to create a CSS wave effect for a hero section in a React application. Begin by designing a visually appealing wave pattern that can serve as a background for the hero page. Ensure that the wave is responsive and adapts well to different screen sizes. Utilize CSS animations to enhance the visual dynamics of the wave, making it engaging for users. When implementing this in React, consider using styled-components or CSS modules for better organization and maintainability of your styles. Be cautious of browser compatibility issues and test the wave effect across various browsers to ensure consistent performance. The final output should be a clean and organized code snippet that can be easily integrated into a React component, along with any necessary comments to explain the functionality of the code.", ),
    ("help w frankenstein fishbowl ap lit give evidence",
     """You are tasked with facilitating a discussion in a fishbowl format for an Advanced Placement Literature class, focusing on the book "Frankenstein" by Mary Shelley. Your objective is to identify and present key pieces of evidence from the text that can be used to support various themes, character analyses, or plot developments during the discussion. Ensure that the evidence you select is relevant and can provoke thoughtful dialogue among participants. Consider significant quotes, character actions, and pivotal moments in the narrative that illustrate the central ideas of the book. When presenting your evidence, structure it clearly, providing context for each piece and explaining its significance to the overall themes of "Frankenstein." Aim to engage your peers in meaningful conversation by encouraging them to respond to the evidence you present."""),
    ("help w frankenstein fishbowl ap lit give evidence",
     """You are tasked with facilitating a discussion in a fishbowl format for an Advanced Placement Literature class, focusing on the book "Frankenstein" by Mary Shelley. Your objective is to identify and present key pieces of evidence from the text that can be used to support various themes, character analyses, or plot developments during the discussion. Ensure that the evidence you select is relevant and can provoke thoughtful dialogue among participants. Consider significant quotes, character actions, and pivotal moments in the narrative that illustrate the central ideas of the book. When presenting your evidence, structure it clearly, providing context for each piece and explaining its significance to the overall themes of "Frankenstein." Aim to engage your peers in meaningful conversation by encouraging them to respond to the evidence you present."""),
    ("help w frankenstein fishbowl ap lit give evidence",
     """You are tasked with facilitating a discussion in a fishbowl format for an Advanced Placement Literature class, focusing on the book "Frankenstein" by Mary Shelley. Your objective is to identify and present key pieces of evidence from the text that can be used to support various themes, character analyses, or plot developments during the discussion. Ensure that the evidence you select is relevant and can provoke thoughtful dialogue among participants. Consider significant quotes, character actions, and pivotal moments in the narrative that illustrate the central ideas of the book. When presenting your evidence, structure it clearly, providing context for each piece and explaining its significance to the overall themes of "Frankenstein." Aim to engage your peers in meaningful conversation by encouraging them to respond to the evidence you present."""),
    ("help w frankenstein fishbowl ap lit give evidence",
     """You are tasked with facilitating a discussion in a fishbowl format for an Advanced Placement Literature class, focusing on the book "Frankenstein" by Mary Shelley. Your objective is to identify and present key pieces of evidence from the text that can be used to support various themes, character analyses, or plot developments during the discussion. Ensure that the evidence you select is relevant and can provoke thoughtful dialogue among participants. Consider significant quotes, character actions, and pivotal moments in the narrative that illustrate the central ideas of the book. When presenting your evidence, structure it clearly, providing context for each piece and explaining its significance to the overall themes of "Frankenstein." Aim to engage your peers in meaningful conversation by encouraging them to respond to the evidence you present."""),
    # add more prompt pairs here...
]

results = run_ab_test(prompt_pairs)
analysis = analyze_results(results)

# Optionally save results for further analysis
with open("prompt_ab_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
