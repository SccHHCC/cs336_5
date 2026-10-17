"""Run all five learner functions in a tiny CPU training loop."""

import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "demo", Path(__file__).with_name("06_sft_your_functions.py")
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
practice = demo.practice


class TeachingTokenizer:
    """A fixed teaching vocabulary; not a pretrained model tokenizer."""
    pad_token_id = 0
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        vocabulary = {"Q": 1, ":": 2, "A": 3, "B": 4, "C": 5, "D": 6}
        return [vocabulary[character] for character in text]


def main():
    torch.manual_seed(42)
    model = demo.TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    batch = practice.tokenize_prompt_and_output(
        prompt_strs=["Q:", "Q:"],
        output_strs=["ABC", "D"],
        tokenizer=TeachingTokenizer(),
    )
    print("Input IDs:\n", batch["input_ids"])
    print("Labels:\n", batch["labels"])
    print("Response mask:\n", batch["response_mask"])

    # Assignment convention: token loss sum / fixed C / sample count / K.
    # C=2, B=2, K=1 gives denominator 4 for this particular batch.
    normalize_constant = 2.0
    for update in range(1, 4):
        optimizer.zero_grad()
        outputs = practice.get_response_log_probs(
            model, batch["input_ids"], batch["labels"], return_token_entropy=True
        )
        loss, metadata = practice.sft_microbatch_train_step(
            policy_log_probs=outputs["log_probs"],
            response_mask=batch["response_mask"],
            gradient_accumulation_steps=1,
            normalize_constant=normalize_constant,
        )
        # The helper already called backward. Do not call backward again here.
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(
            f"Update {update}: loss before update={loss.item():.6f}, "
            f"gradient norm before clipping={gradient_norm.item():.6f}"
        )

    with torch.no_grad():
        outputs = practice.get_response_log_probs(
            model, batch["input_ids"], batch["labels"]
        )
        final_loss = practice.masked_normalize(
            -outputs["log_probs"], batch["response_mask"], normalize_constant
        ) / batch["input_ids"].shape[0]
    print(f"Loss after all three updates: {final_loss.item():.6f}")
    print("Teaching example only: no validation accuracy was measured.")


if __name__ == "__main__":
    main()
