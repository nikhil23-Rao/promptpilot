from collections import Counter
import os
from litellm import completion
import math


def call_llm(prompt):
    return completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        temperature=0
    )


def get_new_prompt(original_prompt):
    system_message = (
        "You are an expert prompt engineer specializing in writing prompts that "
        "maximize LLM output confidence (i.e., higher average token probability). "
        "Rewrite the user's prompt so that the LLM's output will be more confident, "
        "clear, concise, and unambiguous."

        "examples:"
        "Original: condicional spanish ireegulars\n"
        "return: 'Identify and list the irregular verbs in Spanish's conditional tense, providing examples for each irregular verb.'\n\n"
    )

    user_message = (
        f"Original prompt:\n{original_prompt}\n\n"
        "Rewrite this prompt to maximize the LLM's confidence when answering. "
        "Don't remove/rephrase text if a user provides a blurb,article, or any external text it wants the LLM to read."
        "Make it clear, specific, and simple. Feel free to add any information that would make the prompt more clear. **RETURN THE STRING OF THE OPTIMIZED PROMPT ONLY**."
    )

    new_prompt = completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return new_prompt.choices[0].message.content


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


def optimize_prompt_confidence(original_prompt, max_attempts=3, weights=None):
    if weights is None:
        # Lower entropy = more confidence
        weights = {"logprob": 1.0, "entropy": -0.5}

    og_res = completion(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=250,
        messages=[{"role": "user", "content": original_prompt}],
        logprobs=True,
    )
    og_avg = math.exp(
        get_average_logprob_from_modelresponse(og_res))

    best_prompt = original_prompt
    best_score = og_avg

    for attempt in range(max_attempts):
        new_prompt = get_new_prompt(best_prompt)

        # Generate response to new prompt
        response = completion(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=250,
            messages=[{"role": "user", "content": new_prompt}],
            logprobs=True,
        )

        avg_logprob = math.exp(
            get_average_logprob_from_modelresponse(response))

        print("PROB", avg_logprob, og_avg)

        total_score = avg_logprob

        print(
            f"Attempt {attempt+1}: logprob={avg_logprob:.3f}")

        if total_score > best_score:
            best_score = total_score
            best_prompt = new_prompt
            return best_prompt, best_score

    return best_prompt, best_score


def autocomplete(prompt):
    response = completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Autocomplete: {prompt}"}
        ],
        max_tokens=25,
        temperature=0.7,
        logprobs=True,  # Useful if you want top token probabilities
        stream=False
    )
    return response.choices[0].message.content


print(optimize_prompt_confidence(
    """whats coppilot vscoce""", 3)
)
