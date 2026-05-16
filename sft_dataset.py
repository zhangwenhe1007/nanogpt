import argparse
import json
import random

from datasets import load_dataset


END = "<|endoftext|>"


def clean(text):
    return " ".join(str(text or "").split())


def load_stream(dataset_name, *, config=None, split="train"):
    if config is None:
        return iter(load_dataset(dataset_name, split=split, streaming=True))
    return iter(load_dataset(dataset_name, config, split=split, streaming=True))


def make_source(name, bucket, weight, dataset_name, formatter, *, config=None, split="train"):
    try:
        iterator = load_stream(dataset_name, config=config, split=split)
    except Exception as exc:
        print(f"skipping {name}: failed to load {dataset_name}: {exc}")
        return None

    return {
        "name": name,
        "bucket": bucket,
        "weight": weight,
        "dataset_name": dataset_name,
        "config": config,
        "split": split,
        "iterator": iterator,
        "formatter": formatter,
        "resets": 0,
    }


def reset_source(source):
    source["iterator"] = load_stream(
        source["dataset_name"],
        config=source["config"],
        split=source["split"],
    )
    source["resets"] += 1


def trim(text, max_chars):
    text = clean(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def make_qa(question, answer, source, *, context=None, max_prompt_chars=3000, max_response_chars=2000):
    question = clean(question)
    answer = trim(answer, max_response_chars)
    context = trim(context, max_prompt_chars) if context else ""

    if not question or not answer:
        return None

    if context:
        prompt = (
            "### Instruction:\nAnswer the question using the given context.\n\n"
            "### Input:\nContext: " + context + "\nQuestion: " + question + "\n\n"
            "### Response:\n"
        )
        return {"type": "qa", "source": source, "prompt": prompt, "response": answer}

    prompt = "Question: " + question + "\nAnswer: "
    return {"type": "qa", "source": source, "prompt": prompt, "response": answer}


def make_instruction(instruction, response, source, *, input_text=None, max_prompt_chars=3000, max_response_chars=2000):
    instruction = trim(instruction, max_prompt_chars)
    response = trim(response, max_response_chars)
    input_text = trim(input_text, max_prompt_chars) if input_text else ""

    if not instruction or not response:
        return None

    if input_text:
        prompt = (
            "### Instruction:\n" + instruction + "\n\n"
            "### Input:\n" + input_text + "\n\n"
            "### Response:\n"
        )
    else:
        prompt = "### Instruction:\n" + instruction + "\n\n### Response:\n"

    return {"type": "instruction", "source": source, "prompt": prompt, "response": response}


def make_chat(messages, source, *, max_prompt_chars=3000, max_response_chars=2000):
    cleaned = []
    for msg in messages:
        role = msg.get("role", "")
        content = clean(msg.get("content", ""))
        if role and content:
            cleaned.append({"role": role, "content": content})

    if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
        return None

    response = trim(cleaned[-1]["content"], max_response_chars)
    if not response:
        return None

    prompt_parts = []
    for msg in cleaned[:-1]:
        role = msg["role"]
        content = trim(msg["content"], max_prompt_chars)
        if role in {"user", "prompter", "human"}:
            prompt_parts.append("### User:\n" + content)
        elif role in {"assistant", "gpt"}:
            prompt_parts.append("### Assistant:\n" + content)

    if not prompt_parts:
        return None

    prompt = "\n\n".join(prompt_parts) + "\n\n### Response:\n"
    prompt = trim(prompt, max_prompt_chars)
    if not prompt.endswith("### Response:"):
        prompt = prompt + "\n\n### Response:"
    prompt = prompt + "\n"

    return {"type": "chat", "source": source, "prompt": prompt, "response": response, "messages": cleaned}


def render_text(record):
    return record["prompt"] + record["response"] + END


def render_jsonl(record):
    out = dict(record)
    out["text"] = render_text(record)
    return json.dumps(out, ensure_ascii=False)


def format_squad(ex):
    answers = ex.get("answers", {})
    answer_list = answers.get("text", [])
    if not answer_list:
        return None
    return make_qa(ex.get("question", ""), answer_list[0], "squad", context=ex.get("context", ""))


def format_natural_questions_clean(ex):
    question = ex.get("question") or ex.get("query") or ""
    answer = ex.get("answer") or ex.get("answers") or ex.get("long_answer") or ex.get("short_answer") or ""
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return make_qa(question, answer, "natural_questions")


def format_triviaqa(ex):
    ans = ex.get("answer", "")
    if isinstance(ans, dict):
        answer = ans.get("value") or ans.get("normalized_value")
        if answer is None and ans.get("aliases"):
            answer = ans["aliases"][0]
    else:
        answer = str(ans)
    return make_qa(ex.get("question", ""), answer, "triviaqa")


def answer_from_choices(choices, answer_key):
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    for label, text in zip(labels, texts):
        if label == answer_key:
            return text
    return answer_key


def format_openbookqa(ex):
    answer = answer_from_choices(ex.get("choices", {}), ex.get("answerKey", ""))
    return make_qa(ex.get("question_stem") or ex.get("question", ""), answer, "openbookqa")


def format_arc(ex):
    answer = answer_from_choices(ex.get("choices", {}), ex.get("answerKey", ""))
    return make_qa(ex.get("question", ""), answer, "arc")


def format_boolq(ex):
    answer = "yes" if ex.get("answer") else "no"
    question = ex.get("question", "")
    passage = ex.get("passage", "")
    return make_qa(question, answer, "boolq", context=passage)


def format_gsm8k(ex):
    return make_qa(ex.get("question", ""), ex.get("answer", ""), "gsm8k")


def format_dolly(ex):
    return make_instruction(
        ex.get("instruction", ""),
        ex.get("response", ""),
        "dolly",
        input_text=ex.get("context", ""),
    )


def format_alpaca(ex):
    return make_instruction(
        ex.get("instruction", ""),
        ex.get("output", ""),
        "alpaca",
        input_text=ex.get("input", ""),
    )


def format_openorca(ex):
    system_prompt = ex.get("system_prompt", "")
    question = ex.get("question", "")
    if system_prompt:
        instruction = system_prompt + "\n\n" + question
    else:
        instruction = question
    return make_instruction(instruction, ex.get("response", ""), "openorca")


def format_ultrachat(ex):
    messages = ex.get("messages", [])
    return make_chat(messages, "ultrachat")


def format_lmsys(ex):
    if ex.get("redacted"):
        return None
    messages = ex.get("conversation", [])
    return make_chat(messages, "lmsys")


def format_wildchat(ex):
    if ex.get("redacted") or ex.get("toxic"):
        return None
    messages = ex.get("conversation", [])
    return make_chat(messages, "wildchat")


def source_specs(mode):
    qa = [
        ("squad", "qa", 0.35, "rajpurkar/squad", format_squad, None, "train"),
        ("natural_questions", "qa", 0.25, "rojagtap/natural_questions_clean", format_natural_questions_clean, None, "train"),
        ("triviaqa", "qa", 0.15, "mandarjoshi/trivia_qa", format_triviaqa, "rc.nocontext", "train"),
        ("openbookqa", "qa", 0.10, "allenai/openbookqa", format_openbookqa, "main", "train"),
        ("arc_easy", "qa", 0.05, "allenai/ai2_arc", format_arc, "ARC-Easy", "train"),
        ("arc_challenge", "qa", 0.05, "allenai/ai2_arc", format_arc, "ARC-Challenge", "train"),
        ("boolq", "qa", 0.05, "google/boolq", format_boolq, None, "train"),
    ]

    instruction = [
        ("openorca", "instruction", 0.50, "Open-Orca/OpenOrca", format_openorca, None, "train"),
        ("dolly", "instruction", 0.20, "databricks/databricks-dolly-15k", format_dolly, None, "train"),
        ("alpaca", "instruction", 0.20, "tatsu-lab/alpaca", format_alpaca, None, "train"),
        ("gsm8k_instruction", "instruction", 0.10, "openai/gsm8k", format_gsm8k, "main", "train"),
    ]

    chat = [
        ("ultrachat", "chat", 0.70, "HuggingFaceH4/ultrachat_200k", format_ultrachat, None, "train_sft"),
        ("lmsys", "chat", 0.20, "lmsys/lmsys-chat-1m", format_lmsys, None, "train"),
        ("wildchat", "chat", 0.10, "allenai/WildChat-1M", format_wildchat, None, "train"),
    ]

    math = [
        ("gsm8k_math", "math", 0.70, "openai/gsm8k", format_gsm8k, "main", "train"),
        ("openbookqa_math", "math", 0.15, "allenai/openbookqa", format_openbookqa, "main", "train"),
        ("arc_challenge_math", "math", 0.15, "allenai/ai2_arc", format_arc, "ARC-Challenge", "train"),
    ]

    if mode == "qa":
        return qa
    if mode == "instruction":
        return instruction
    if mode == "chat":
        return chat
    if mode == "mixed":
        return scale_bucket(qa, 0.40) + scale_bucket(instruction, 0.30) + scale_bucket(chat, 0.20) + scale_bucket(math, 0.10)
    raise ValueError(f"unknown mode: {mode}")


def scale_bucket(specs, bucket_weight):
    total = sum(spec[2] for spec in specs)
    return [
        (name, bucket, weight / total * bucket_weight, dataset, formatter, config, split)
        for name, bucket, weight, dataset, formatter, config, split in specs
    ]


def build_sources(mode):
    sources = []
    for name, bucket, weight, dataset_name, formatter, config, split in source_specs(mode):
        source = make_source(
            name,
            bucket,
            weight,
            dataset_name,
            formatter,
            config=config,
            split=split,
        )
        if source is not None:
            sources.append(source)

    if not sources:
        raise RuntimeError("no sources loaded")

    total = sum(source["weight"] for source in sources)
    for source in sources:
        source["weight"] = source["weight"] / total
    return sources


def next_record(source):
    while True:
        try:
            ex = next(source["iterator"])
        except StopIteration:
            reset_source(source)
            ex = next(source["iterator"])

        record = source["formatter"](ex)
        if record:
            return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="mixed", choices=["qa", "instruction", "chat", "mixed"])
    parser.add_argument("--output", type=str, default="sft_mixed_1m.txt")
    parser.add_argument("--num-examples", type=int, default=1_000_000)
    parser.add_argument("--format", type=str, default="text", choices=["text", "jsonl"])
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    sources = build_sources(args.mode)
    names = [source["name"] for source in sources]
    weights = [source["weight"] for source in sources]
    by_name = {source["name"]: source for source in sources}
    counts = {name: 0 for name in names}
    buckets = {source["bucket"]: 0 for source in sources}
    resets = {name: 0 for name in names}

    renderer = render_text if args.format == "text" else render_jsonl

    print("sources:")
    for source in sources:
        print(f"  {source['name']}: bucket={source['bucket']} weight={source['weight']:.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_examples):
            name = random.choices(names, weights=weights, k=1)[0]
            source = by_name[name]
            record = next_record(source)

            f.write(renderer(record))
            f.write("\n")

            counts[name] += 1
            buckets[source["bucket"]] += 1
            resets[name] = source["resets"]

            if (i + 1) % 10_000 == 0:
                print(f"wrote {i + 1} examples")
                print(f"  buckets: {buckets}")
                print(f"  sources: {counts}")
                print(f"  resets: {resets}")

    print(f"done: wrote {args.num_examples} examples to {args.output}")
    print(f"buckets: {buckets}")
    print(f"sources: {counts}")
    print(f"resets: {resets}")


if __name__ == "__main__":
    main()
