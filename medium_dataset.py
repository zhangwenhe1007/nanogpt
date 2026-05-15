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


def format_plain_text(ex):
    text = ex.get("text") or ex.get("content") or ""
    text = clean(text)
    if not text:
        return None
    return text + END


def format_tinystory(ex):
    text = clean(ex["text"])
    if not text:
        return None
    return text + END


def format_gsm8k(ex):
    return (
        "Question: " + clean(ex["question"]) + "\n"
        "Answer: " + clean(ex["answer"]) + END
    )


def format_trivia(ex):
    ans = ex["answer"]
    if isinstance(ans, dict):
        answer = ans.get("value") or ans.get("normalized_value")
        if answer is None and ans.get("aliases"):
            answer = ans["aliases"][0]
    else:
        answer = str(ans)

    if not answer:
        return None

    return (
        "Question: " + clean(ex["question"]) + "\n"
        "Answer: " + clean(answer) + END
    )


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

    return (
        "Question: " + clean(ex["question_stem"]) + "\n"
        "Answer: " + clean(answer) + END
    )


def next_formatted(iterator, formatter):
    while True:
        formatted = formatter(next(iterator))
        if formatted:
            return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="medium_train.txt")
    parser.add_argument("--num-examples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    sources = [
        {
            "name": "fineweb_edu",
            "weight": 0.60,
            "iterator": load_stream("HuggingFaceFW/fineweb-edu", config="sample-10BT"),
            "formatter": format_plain_text,
        },
        {
            "name": "cosmopedia",
            "weight": 0.20,
            "iterator": load_stream("HuggingFaceTB/cosmopedia"),
            "formatter": format_plain_text,
        },
        {
            "name": "openwebmath",
            "weight": 0.10,
            "iterator": load_stream("open-web-math/open-web-math"),
            "formatter": format_plain_text,
        },
        {
            "name": "tinystories",
            "weight": 0.05,
            "iterator": load_stream("roneneldan/TinyStories"),
            "formatter": format_tinystory,
        },
        {
            "name": "gsm8k",
            "weight": 0.025,
            "iterator": load_stream("openai/gsm8k", config="main"),
            "formatter": format_gsm8k,
        },
        {
            "name": "triviaqa",
            "weight": 0.015,
            "iterator": load_stream("mandarjoshi/trivia_qa", config="rc.nocontext"),
            "formatter": format_trivia,
        },
        {
            "name": "openbookqa",
            "weight": 0.01,
            "iterator": load_stream("allenai/openbookqa", config="main"),
            "formatter": format_openbookqa,
        },
    ]

    names = [source["name"] for source in sources]
    weights = [source["weight"] for source in sources]
    by_name = {source["name"]: source for source in sources}
    counts = {name: 0 for name in names}

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_examples):
            name = random.choices(names, weights=weights, k=1)[0]
            source = by_name[name]
            text = next_formatted(source["iterator"], source["formatter"])

            f.write(text)
            f.write("\n")
            counts[name] += 1

            if (i + 1) % 10_000 == 0:
                print(f"wrote {i + 1} examples: {counts}")

    print(f"done: wrote {args.num_examples} examples to {args.output}")
    print(counts)


if __name__ == "__main__":
    main()
