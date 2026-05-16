import argparse
import random

from datasets import load_dataset


END = "<|endoftext|>"


def clean(text):
    return " ".join(str(text).split())


def load_stream(dataset_name, *, config=None, split="train"):
    if config is None:
        return iter(load_dataset(dataset_name, split=split, streaming=True))
    return iter(load_dataset(dataset_name, config, split=split, streaming=True))


def make_source(name, weight, dataset_name, formatter, *, config=None, split="train"):
    return {
        "name": name,
        "weight": weight,
        "dataset_name": dataset_name,
        "config": config,
        "split": split,
        "iterator": load_stream(dataset_name, config=config, split=split),
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


def qa_format(question, answer):
    question = clean(question)
    answer = clean(answer)
    if not question or not answer:
        return None

    return (
        "Question: " + question + "\n"
        "Answer: " + answer + END
    )


def instruction_format(instruction, response, input_text=None):
    instruction = clean(instruction)
    response = clean(response)
    input_text = clean(input_text or "")

    if not instruction or not response:
        return None

    if input_text:
        return (
            "### Instruction:\n" + instruction + "\n\n"
            "### Input:\n" + input_text + "\n\n"
            "### Response:\n" + response + END
        )

    return (
        "### Instruction:\n" + instruction + "\n\n"
        "### Response:\n" + response + END
    )


def format_gsm8k(ex):
    return qa_format(ex["question"], ex["answer"])


def format_trivia(ex):
    ans = ex["answer"]
    if isinstance(ans, dict):
        answer = ans.get("value") or ans.get("normalized_value")
        if answer is None and ans.get("aliases"):
            answer = ans["aliases"][0]
    else:
        answer = str(ans)

    return qa_format(ex["question"], answer)


def format_openbookqa(ex):
    choices = ex.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    answer_key = ex.get("answerKey")

    answer = None
    for label, text in zip(labels, texts):
        if label == answer_key:
            answer = text
            break

    if answer is None:
        answer = answer_key

    question = ex.get("question_stem") or ex.get("question") or ""
    return qa_format(question, answer)


def format_squad(ex):
    answers = ex.get("answers", {})
    answer_list = answers.get("text", [])
    if not answer_list:
        return None

    question = ex.get("question", "")
    context = ex.get("context", "")
    answer = answer_list[0]

    return instruction_format(
        "Answer the question using the given context.",
        answer,
        "Context: " + context + "\nQuestion: " + question,
    )


def format_dolly(ex):
    return instruction_format(
        ex.get("instruction", ""),
        ex.get("response", ""),
        ex.get("context", ""),
    )


def next_formatted(source):
    while True:
        try:
            ex = next(source["iterator"])
        except StopIteration:
            reset_source(source)
            ex = next(source["iterator"])

        formatted = source["formatter"](ex)
        if formatted:
            return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="instruction_train.txt")
    parser.add_argument("--num-examples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    sources = [
        make_source("dolly", 0.35, "databricks/databricks-dolly-15k", format_dolly),
        make_source("squad", 0.25, "rajpurkar/squad", format_squad),
        make_source("gsm8k", 0.20, "openai/gsm8k", format_gsm8k, config="main"),
        make_source("triviaqa", 0.15, "mandarjoshi/trivia_qa", format_trivia, config="rc.nocontext"),
        make_source("openbookqa", 0.05, "allenai/openbookqa", format_openbookqa, config="main"),
    ]

    names = [source["name"] for source in sources]
    weights = [source["weight"] for source in sources]
    by_name = {source["name"]: source for source in sources}
    counts = {name: 0 for name in names}
    resets = {name: 0 for name in names}

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_examples):
            name = random.choices(names, weights=weights, k=1)[0]
            source = by_name[name]
            text = next_formatted(source)

            f.write(text)
            f.write("\n")
            counts[name] += 1
            resets[name] = source["resets"]

            if (i + 1) % 10_000 == 0:
                print(f"wrote {i + 1} examples: {counts}; resets: {resets}")

    print(f"done: wrote {args.num_examples} examples to {args.output}")
    print(counts)
    print(f"resets: {resets}")


if __name__ == "__main__":
    main()
