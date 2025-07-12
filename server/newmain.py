import math
from numpy import dot
from numpy.linalg import norm
from collections import Counter
import os
from litellm import completion, embedding
from flask import Flask, request, jsonify

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
        max_tokens=400,
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
            max_tokens=400,
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


@app.route("/optimize", methods=["POST"])
def api_optimize():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_attempts = int(data.get("max_attempts", 1))
    best_prompt, best_score = optimize_prompt_multifactors(
        prompt, max_attempts)
    return jsonify({"optimized_prompt": best_prompt, "confidence": best_score})


# @app.route("/autocomplete", methods=["POST"])
# def api_autocomplete():
#     data = request.get_json(force=True)
#     prompt = data.get("prompt", "")
#     suggestion = autocomplete(prompt)
#     return jsonify({"autocomplete": suggestion})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1001)
