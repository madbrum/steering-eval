import numpy as np
import string
from collections import Counter, defaultdict
import nltk
import scipy
import re
from parlai.utils.safety import OffensiveLanguageClassifier
from transformers import pipeline
nltk.download("punkt")
nltk.download("punkt_tab")

dataset_name = ""
toxicity_classifier = None
sentiment_classifier = None


def send_dataset_name(ds_name):
    """
    Loads toxicity/sentiment classifiers if needed for current task.
    """
    global dataset_name 
    global toxicity_classifier
    global sentiment_classifier
    dataset_name = ds_name
    if dataset_name == "paradetox" or dataset_name == "detoxification":
        toxicity_classifier = OffensiveLanguageClassifier(custom_model_file="zoo:bot_adversarial_dialogue/multi_turn/model")
    if dataset_name == "sentiment_transfer":
        sentiment_classifier = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english", device="cuda")


def generate(sentence, model, tokenizer):
    """
    Generate model completion to given sentence. 
    
    Args:
        sentence (str): The prompt to generate completion to.
        model (GenerationMixin): Generation-capable Huggingface language model.
        tokenizer (TokenizerBackend): The model tokenizer. 
    
    Returns:
        decoded model generation (str)
    """
    MAX_NEW_TOKENS = 16
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                            max_new_tokens=MAX_NEW_TOKENS)
    decoded_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
    return decoded_output


def batched_generate(batch, model, tokenizer):
    """
    Generate model completions to given batch of sentences. 
    
    Args:
        batch (list[str]): The batch of prompts to generate completions to.
        model (GenerationMixin): Generation-capable Huggingface language model.
        tokenizer (TokenizerBackend): The model tokenizer. 
    
    Returns:
        batch of decoded model generations (list[str])
    """
    MAX_NEW_TOKENS = 16
    inputs = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, top_p=0.9, temperature=0.1,
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded_outputs = tokenizer.batch_decode([x[-MAX_NEW_TOKENS:] for x in outputs])
    return decoded_outputs


def evaluate(output, target, metrics, scores, parsed):
    """
    Evaluates a text output across given metrics.
    
    Args: 
        output (str): The text to evaluate.
        target (str): The ground truth text.
        metrics (list[str]): The list of metric names.
        scores (dict[str, list]): Map from metric name of list of scores.
        parsed (list[str]): list of parsed strings 
    """
    parsed_str = None
    for metric in metrics:
        score = 0
        if metric in ground_truth_metrics:
            parsed_str, score = parse_generation(output, [target], metric_name_to_lambda[metric])
        else:
            score = metric_name_to_lambda[metric](output)
            if metric == "distinct" and all(s == 0 for s in score):
                score = [0] * len(score)  # Handle case where distinct returns all zeros
        scores[metric].append(score)
    if parsed_str is not None:
        parsed.append(parsed_str)


def avg_scores(scores):
    """
    Averages score lists for each metric.

    Args:
        scores (dict[str, list]): Map from metric name of list of scores.

    Returns:
        dict[str, float] or dict[str, list[float]]: Map from metric name to averaged score,
         or map from metric name to list of averaged scores if metric is "distinct".
    """
    avgs = defaultdict(lambda: 0)
    for key in scores.keys():
        if key == "distinct":
            avgs[key] = list(np.mean(scores[key], axis=0))
        else:
            avgs[key] = np.mean(scores[key])
    return avgs


def normalize_answer(s):
    """
    Lowercase text and remove punctuation, articles and extra whitespace.
    From https://github.com/huggingface/evaluate
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        if "capitalize" in dataset_name:
            return text
        else:
            return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Pick maximum score across possible answers.
    From https://github.com/ericwtodd/function_vectors
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def parse_generation(output_str, target, metric_fn):
    """
    Parse a generated string for the target, and score using the specified metric.
    From https://github.com/ericwtodd/function_vectors
    """
    ans_regex = re.compile("([\w. ]+)[\nQ]*")
    parsed_str = ans_regex.findall(output_str)
    if len(parsed_str) > 0:
        parsed_str = parsed_str[0]
        score = metric_max_over_ground_truths(metric_fn, parsed_str, target) if metric_fn != present_score \
            else metric_max_over_ground_truths(metric_fn, output_str, target)
    else:
        score = 0.0

    return parsed_str, score


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    """
    Computes weighted average of n-gram entropies of given sentence. 
    Logic from https://github.com/kmeng01/rome. 

    Args:
        sentence (str): The text to evaluate.
        ns (Optional, list[int]): Values of n for n-grams.
        weights (list[float]): Weights of n-gram scores, same size as ns.
        agg (str): Aggregation type, must be "arith" or "geom". Default is "arith".

    Returns:
        float: Weighted average of n-gram entropies.
    """
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    """
    Computes frequency of n-grams in given sentence.
    Logic from https://github.com/kmeng01/rome.

    Args:
        sentence (str): The text to evaluate.
        n (int): The value of n for n-grams. Default is 2.

    Returns:
        nltk.FreqDist: Frequency distribution of n-grams.
    """
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def first_word_score(prediction, ground_truth):
    """
    Computes first word score of given prediction and ground truth
    
    Args:
        prediction (str): The predicted text.
        ground_truth (str): The ground truth text.

    Returns:
        int: 1 if first word matches, 0 o.w.
    """
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth).split()
    if len(prediction) > 0 and len(ground_truth) > 0:
        return 1 if prediction[0] == ground_truth[0] else 0
    else:
        return 1 if prediction == ground_truth else 0


def f1_score(prediction, ground_truth):
    """
    Computes F1 score of given prediction and ground truth
    
    Args:
        prediction (str): The predicted text.
        ground_truth (str): The ground truth text.

    Returns:
        float: F1 score.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def present_score(prediction, ground_truth):
    """
    Computes present score of given prediction and ground truth
    
    Args:
        prediction (str): The predicted text.
        ground_truth (str): The ground truth text.

    Returns:
        int: 1 if ground truth appears anywhere in prediction, 0 o.w.
    """
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth)
    if len(prediction) > 0:
        return 1 if ground_truth in prediction else 0
    else:
        return 1 if ground_truth == prediction else 0


def distinct(text, ns=[1, 2]):
    """
    Computes Dist-n scores of given text, or weighted counts ofunique n-grams in text.
    
    Args:
        text (str): The text to evaluate.
        ns (list[int]): The values of n for Dist-n scores. Default is [1, 2].

    Returns:
        list[float]: List of Dist-n scores.
    """
    tokens = nltk.word_tokenize(text)
    if len(tokens) == 0:
        # Return a list of zeros with the same length as ns
        return [0] * len(ns)
    dists = []
    for n in ns:
        distinct_ngrams = list(set(nltk.ngrams(tokens, n)))
        dists.append(len(distinct_ngrams) / len(tokens))
    return dists


def toxicity_score(text):
    """
    Computes toxicity score of given text.
    
    Args:
        text (str): The text to evaluate.

    Returns:
        int: 1 if classifier identifies toxicity with 90% chance, 0 o.w.
    """
    ppred, prob = toxicity_classifier.contains_offensive_language(text)
    return 1 if prob > 0.9 and ppred else 0


def sentiment_score(output):
    """ 
    Computes sentiment score of given text.
    
    Args:
        output (str): The text to evaluate.

    Returns:
        int: 1 if output sentiment is positive, 0 o.w.
    """
    #success if output sentiment is positive, since we look for negative -> positive sentiment transfer
    output_label = sentiment_classifier(output)[0]["label"]
    return 1 if output_label == "POSITIVE" else 0


metric_name_to_lambda = {
    "first_word_score" : first_word_score,
    "f1_score": f1_score,
    #entropy is bad metric if there are no n_grams > 2
    "n_gram_entropy": compute_n_gram_entropy,
    "distinct": distinct,
    "present_score": present_score,
    "toxicity_score": toxicity_score,
    "sentiment_score": sentiment_score
}

ground_truth_metrics = ["first_word_score", "f1_score", "present_score"]

