"""CPU lesson: inspect local Qwen tokenization, padding, and label alignment."""

from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model_dir = Path(__file__).resolve().parents[1] / "model" / "Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    prompts = ["Question: 3 + 2 = ?\nAnswer:", "Question: 1 + 1 = ?\nAnswer:"]
    responses = ["The answer is 5.", "2."]

    sequences = []
    masks = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        sequences.append(prompt_ids + response_ids)
        masks.append([0] * len(prompt_ids) + [1] * len(response_ids))
        print("\nprompt:", repr(prompt))
        print("prompt_ids:", prompt_ids)
        print("response:", repr(response))
        print("response_ids:", response_ids)

    max_len = max(map(len, sequences))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    padded_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    padded_masks = torch.zeros_like(padded_ids)
    for row, (ids, mask) in enumerate(zip(sequences, masks)):
        padded_ids[row, :len(ids)] = torch.tensor(ids)
        padded_masks[row, :len(mask)] = torch.tensor(mask)

    input_ids = padded_ids[:, :-1]
    labels = padded_ids[:, 1:].clone()
    response_mask = padded_masks[:, 1:]
    print("\npad_token_id:", pad_id)
    print("input_ids:\n", input_ids)
    print("labels:\n", labels)
    print("response_mask:\n", response_mask)
    print("shape:", tuple(input_ids.shape))

    for row in range(len(sequences)):
        print(f"\nSample {row + 1}: position | input -> target | mask")
        for pos in range(input_ids.shape[1]):
            source = tokenizer.convert_ids_to_tokens(int(input_ids[row, pos]))
            target = tokenizer.convert_ids_to_tokens(int(labels[row, pos]))
            print(f"{pos:2d} | {source!r} -> {target!r} | {int(response_mask[row, pos])}")
        selected_ids = labels[row][response_mask[row].bool()].tolist()
        print("Selected response:", repr(tokenizer.decode(selected_ids)))


if __name__ == "__main__":
    main()
