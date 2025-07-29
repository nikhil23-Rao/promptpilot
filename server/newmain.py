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
        max_tokens=200,
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
            max_tokens=200,
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


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import binned_statistic


# m1_conf =[0.6299240467647439, 0.9671454795379872, 0.672439920187527, 0.9674668249143507, 0.9900658467509845, 0.6772084673662162, 0.6053799846061376, 0.9524696542296565, 0.9999963198735957, 0.8560170695873778, 0.9767836092699713, 0.5026816249623436, 0.997732851293215, 0.9854696894352905, 0.6617113736238547, 0.7744777185415042, 0.9989655042555045, 0.9996163970661681, 0.5936589863450329, 0.806177518018621, 0.9961273170220609, 0.8802588019056752, 0.9624495582478065, 0.9700688700351979, 0.9898542963205189, 0.9950359722479925, 0.9887939245470677, 0.997896459499303, 0.9999975415208362, 0.9046460642364229, 0.884561466902981, 0.4868424367019111, 0.789407906558593, 0.7055335476657224, 0.957294437111533, 0.9998855414211422, 0.9599690718168684, 0.6940578578310672, 0.9997748831494712, 0.5828369932796978, 0.9948812643871741, 0.8273858070168307, 0.9977838059444523, 0.9885969887250127, 0.9966901779536151, 0.5442169598923815, 0.9903050322355866, 0.9999336657339533, 0.6497732295833827, 0.746691995615641, 0.9956536510741061, 0.9915902881058821, 0.985664287522045, 0.7645344226106299, 0.9626058407684394, 0.9764487514915595, 0.4727866466912889, 0.999507659424147, 0.9931850962549917, 0.5226582614209299, 0.8209287880366987, 0.9630371450655548, 0.9027074696805844, 0.8017390080855643, 0.9428987290427105, 0.9998124952776882, 0.9991081563133906, 0.6853956953907486, 0.9577292020560151, 0.9837912471990395, 0.9988626929251283, 0.9945807358711164, 0.7746120860630582, 0.9554730346418558, 0.8775911413407099, 0.9962814800332961, 0.99999503825305, 0.724915670736121, 0.9991880822136244, 0.7998792290974304, 0.920695978518824, 0.721812454731536, 0.9999977055187121, 0.9993241021225499, 0.9180400774139856, 0.9999561788060649, 0.829469390127657, 0.9998275085741326, 0.9624212804454222, 0.9829164687645732, 0.4009718111458346, 0.9808721296923718, 0.9995624830781196, 0.6494193866409277, 0.33126828978307993, 0.9988646391138609, 0.9990769378204976, 0.9978042198735145, 0.6152649133010196, 0.5289522861938821, 0.9953353162736672, 0.4853941989999475, 0.9927956625931416, 0.9560885562390984, 0.9939112715590908, 0.7955276708157955, 0.9919326397209882, 0.995684660200503, 0.9395973172107583, 0.9924010182027954, 0.9998461013008987, 0.5220665365344916, 0.9998632604706298, 0.9650552469210867, 0.7746872090736701, 0.9994459287869206, 0.9997654978328642, 0.8442078685626897, 0.8106834351195564, 0.99999503825305, 0.49707532251032754, 0.5970711306166054, 0.9575907878056313, 0.9347935466520012, 0.9862116116783053, 0.9950802407393736, 0.9969685896308175, 0.997129949775098, 0.9933696280389227, 0.59655647104035, 0.9055525948569854, 0.9960096805404972, 0.9649946138486716, 0.9612748546577462, 0.9254805926066896, 0.9865942555511639, 0.8869148890335957, 0.9940355410377063, 0.7671869150425878, 0.7522955773452721, 0.8880258520215389, 0.9775767554697393, 0.6211520340054719, 0.8711691683872738, 0.39324371368874383, 0.4107990719312862, 0.8560706577626043, 0.8938550911377867, 0.7565937022830994, 0.8789739584204166, 0.9911118913248336, 0.8445819452296923, 0.9676823162871876, 0.5310515403098958, 0.9703707226599577, 0.9491120697338936, 0.42267767959891883, 0.9999992103693378, 0.9085559647336504, 0.7784905262247115, 0.9868095516061448, 0.999926618031652, 0.7783828354413699, 0.5366286018522403, 0.9984533674407002, 0.8793032577948423, 0.9782839126363874, 0.4777133379961549, 0.8843547604797869, 0.871502127847913, 0.5897583303060652, 0.3170139711703695, 0.9422123202882641, 0.4730156669187178, 0.9594383372364159, 0.7768114746935922, 0.9992398547450825, 0.952651000067304, 0.9953323983195844, 0.8814827484981596, 0.5147451083772627, 0.7589157932137951, 0.9976080822411801, 0.67782664298444, 0.9488540187191787, 0.9979337159068141, 0.9999896741288624, 0.9940700414033024, 0.5618518171372243, 0.981714782380258, 0.61573969549215, 0.2730032289977944, 0.5264527815570541, 0.9911370974224925, 0.905334966497564, 0.9990241430197181, 0.9805017441844041, 0.9728318271568782, 0.9820048059749552, 0.9969594197112366]

# m1_acc = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# m2_conf =[0.6499178884570554, 0.924175618685241, 0.0, 0.9833871714807263, 0.9881505462437216, 0.9975344609930397, 0.0, 0.9855647053864613, 0.9999915908828307, 0.9235167123495274, 0.9832512186634303, 0.9265515042085115, 0.9989232851304944, 0.9983441548873833, 0.0, 0.8797653298038374, 0.9981406909071292, 0.9999564267955727, 0.5442581395147139, 0.0, 0.9984493831424244, 0.9994918987302293, 0.9405603886148365, 0.999326538356366, 0.9977756372270316, 0.9996021265074702, 0.9997802956232987, 0.9998384082256874, 0.9999991678813817, 0.9046503807764229, 0.0, 0.7375793312013047, 0.8741244320580245, 0.0, 0.9976871286220969, 0.9999332734305257, 0.0, 0.5960655996541548, 0.9999001093540978, 0.0, 0.9977857333113606, 0.9817126277655851, 0.9959068771021007, 0.9911368303670642, 0.9986906196919056, 0.7876593965513363, 0.9850836792882147, 0.9999504701241914, 0.6197118291160717, 0.8527835071461538, 0.997945385623459, 0.9989280434264229, 0.9962794040438032, 0.9959310279844069, 0.9999644220963291, 0.9740877926103332, 0.8115028305888184, 0.9997690262461488, 0.9985699842430973, 0.9986211521812229, 0.7717158900522157, 0.8773259469564886, 0.0, 0.7783925884109039, 0.9947145137471113, 0.9999191964131283, 0.9996588198001988, 0.0, 0.9960921804905367, 0.9736807283061711, 0.9996282846935751, 0.9985001477150164, 0.0, 0.9710029357582488, 0.9813253143545858, 0.9998340428653559, 0.9999994878280495, 0.7965891563691335, 0.9995319208287936, 0.0, 0.9975018044046583, 0.6779705473034142, 0.9999990213932386, 0.9993480133543078, 0.9051080481854737, 0.999979902674001, 0.8230174756751002, 0.9999516457463516, 0.9626636981291725, 0.9975583715712689, 0.0, 0.9997944098847628, 0.9997905888733973, 0.703932658707069, 0.0, 0.9985605877289587, 0.9998832633699198, 0.9999933979146081, 0.9906029202548596, 0.6375657373054904, 0.9976394560604069, 0.526420368079864, 0.9998498711745585, 0.9975496145820647, 0.9998026074356962, 0.0, 0.9995991650547827, 0.9986219352493227, 0.9399079340186388, 0.9982896199946467, 0.9999836181113373, 0.0, 0.9998818676845072, 0.0, 0.0, 0.9999152527254964, 0.9998756578064497, 0.0, 0.7257657230964527, 0.9999968642920289, 0.8124915740340504, 0.778275238625297, 0.9818275564708602, 0.8904359280874838, 0.9843411564979997, 0.9964234107593496, 0.9998772722299618, 0.9991548898430282, 0.994234623143657, 0.9115978523670665, 0.91100799271027, 0.995950727053828, 0.9961944658505968, 0.9726302202686984, 0.9318079868531235, 0.9983873303377313, 0.9036633873157611, 0.9987906910893096, 0.8268029218868337, 0.0, 0.9268752581430744, 0.9763198531005444, 0.5577406664087715, 0.8448796053278775, 0.6426940600345431, 0.0, 0.6996183348147779, 0.9486861132808583, 0.9737349408751389, 0.0, 0.9911661635333191, 0.0, 0.9934083792313081, 0.6967407904298556, 0.9997143421801569, 0.0, 0.46373679810527146, 0.9999987988199367, 0.9788121485458573, 0.0, 0.9889531497398845, 0.9999287322072657, 0.0, 0.0, 0.9999710992754461, 0.9992494550304124, 0.985716621144807, 0.0, 0.9484931092169073, 0.9368419065657485, 0.9240253452160838, 0.48579730005096333, 0.0, 0.0, 0.9802912380835248, 0.7771637839667087, 0.9999667787825378, 0.9760202635546227, 0.9976453433767785, 0.9619613548278274, 0.6292830756510298, 0.0, 0.9995690638140096, 0.0, 0.9610102017964901, 0.999416955419002, 0.9999947757151217, 0.9961680630499755, 0.896248918590749, 0.9997971439612111, 0.8333479886487922, 0.6836309020119212, 0.9865008856504629, 0.999090677680097, 0.7870010179981658, 0.9996124231466464, 0.0, 0.9499734396654737, 0.9914224684759223, 0.9979119976767552]

# m2_acc =[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


# log_edges = np.logspace(-5, -1, 10)  # 15 bins between 1e-10 and 1e-1
# bin_edges = 1.0 - log_edges[::-1]     # convert to confidence near 1

# bin_labels = [f"{bin_edges[i]:.10f}–{bin_edges[i+1]:.10f}" for i in range(len(bin_edges)-1)]

# def binned_stats(conf, acc, bins):
#     digitized = np.digitize(conf, bins) - 1
#     bin_acc = []
#     bin_conf = []

#     for i in range(len(bins) - 1):
#         idx = (digitized == i)
#         if np.any(idx):
#             bin_acc.append(np.mean(acc[idx]))
#             bin_conf.append(np.mean(conf[idx]))
#         else:
#             bin_acc.append(np.nan)
#             bin_conf.append((bins[i] + bins[i + 1]) / 2)

#     return np.array(bin_conf), np.array(bin_acc)

# # === Compute binned stats
# bin_conf1, bin_acc1 = binned_stats(np.array(m1_conf), np.array(m1_acc), bin_edges)
# bin_conf2, bin_acc2 = binned_stats(np.array(m2_conf), np.array(m2_acc), bin_edges)

# # === Plot
# plt.figure(figsize=(12, 6))
# plt.plot(bin_conf1, bin_acc1, 'bo-', label='baseline')
# plt.plot(bin_conf2, bin_acc2, 'ro-', label='class_proxy')
# plt.plot([0, 1], [0, 1], 'k--', label='perfect calibration')

# # Label points
# for i, label in enumerate(bin_labels):
#     if not np.isnan(bin_acc1[i]):
#         plt.annotate(label, (bin_conf1[i], bin_acc1[i]), textcoords="offset points",
#                      xytext=(0, 8), ha='center', fontsize=7, color='blue')
#     if not np.isnan(bin_acc2[i]):
#         plt.annotate(label, (bin_conf2[i], bin_acc2[i]), textcoords="offset points",
#                      xytext=(0, -12), ha='center', fontsize=7, color='red')

# plt.xlabel('Confidence (zoomed near 1)')
# plt.ylabel('Average Accuracy')
# plt.title('Fine-Grained Calibration Plot: Log Bins Near 1.0')
# plt.legend()
# plt.grid(True)
# plt.xlim(0.97, 1.00001)
# plt.ylim(0, 1.05)
# plt.tight_layout()
# plt.show()


# # # === Define bin edges ===
# # bin_edges = np.linspace(0.0, 1.0, 11)
# # bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

# # def binned_stats(conf, acc, bins):
# #     digitized = np.digitize(conf, bins) - 1
# #     bin_acc = []
# #     bin_conf = []

# #     for i in range(len(bins) - 1):
# #         idx = (digitized == i)
# #         if np.any(idx):
# #             bin_acc.append(np.mean(acc[idx]))
# #             bin_conf.append(np.mean(conf[idx]))
# #         else:
# #             bin_acc.append(np.nan)
# #             bin_conf.append((bins[i] + bins[i + 1]) / 2)

# #     return np.array(bin_conf), np.array(bin_acc)

# # # === Compute binned stats for both models ===
# # bin_conf1, bin_acc1 = binned_stats(np.array(m1_conf), np.array(m1_acc), bin_edges)
# # bin_conf2, bin_acc2 = binned_stats(np.array(m2_conf), np.array(m2_acc), bin_edges)

# # # === Plot both curves ===
# # plt.figure(figsize=(10, 6))
# # plt.plot(bin_conf1, bin_acc1, 'bo-', label='baseline')      # Blue: baseline
# # plt.plot(bin_conf2, bin_acc2, 'ro-', label='class_proxy')   # Red: class_proxy
# # plt.plot([0, 1], [0, 1], 'k--', label='perfect calibration') # Diagonal line

# # # === Annotate dots with bin labels ===
# # for i, label in enumerate(bin_labels):
# #     if not np.isnan(bin_acc1[i]):
# #         plt.annotate(label, (bin_conf1[i], bin_acc1[i]), textcoords="offset points",
# #                      xytext=(0, 8), ha='center', fontsize=8, color='blue')
# #     if not np.isnan(bin_acc2[i]):
# #         plt.annotate(label, (bin_conf2[i], bin_acc2[i]), textcoords="offset points",
# #                      xytext=(0, -12), ha='center', fontsize=8, color='red')

# # # === Labels and styling ===
# # plt.xlabel('Confidence')
# # plt.ylabel('Average Accuracy')
# # plt.title('Accuracy vs Confidence: news2')
# # plt.legend()
# # plt.grid(True)
# # plt.ylim(0, 1.05)
# # plt.xlim(0, 1.0)
# # plt.tight_layout()
# # plt.show()
