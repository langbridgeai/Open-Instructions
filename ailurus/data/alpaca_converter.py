import argparse
import json
import pathlib

PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\n{input}"
    ),
    "prompt_no_input": (
        "{instruction}"
    ),
}


def main(args):
    data_path = pathlib.Path(args.data_path)
    with data_path.open() as f:
        data = json.load(f)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in data
    ]
    targets = [example["output"] for example in data]

    new_data = []
    cnt = 1
    for s, t in zip(sources, targets):
        new_data.append({
            "id": str(cnt),
            "conversations": [
                {
                    "from": "human",
                    "value": s,
                },
                {
                    "from": "gpt",
                    "value": t,
                }
            ]
        })
        cnt += 1

    json.dump(new_data, open(args.output_path, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../open-instructions/alpaca_data.json")
    parser.add_argument("--output_path", type=str, default="../open-instructions/alpaca_td003_en_conv.json")
    args = parser.parse_args()
    main(args)

