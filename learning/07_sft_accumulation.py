"""Compare full-batch and accumulated gradients using the learner's helpers."""

import copy
import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "demo", Path(__file__).with_name("06_sft_your_functions.py")
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def main():
    torch.manual_seed(42)
    full_model = demo.TinyModel()
    micro_model = copy.deepcopy(full_model)
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 6, 0]])
    labels = torch.tensor([[2, 3, 4, 5], [2, 6, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 0]])
    total_tokens = mask.sum()

    full_model.zero_grad()
    outputs = demo.practice.get_response_log_probs(full_model, input_ids, labels)
    full_loss = demo.practice.masked_normalize(
        -outputs["log_probs"], mask, total_tokens
    )
    full_loss.backward()

    micro_model.zero_grad()
    accumulated_loss = 0.0
    for index in range(2):
        outputs = demo.practice.get_response_log_probs(
            micro_model, input_ids[index:index+1], labels[index:index+1]
        )
        loss = demo.practice.masked_normalize(
            -outputs["log_probs"], mask[index:index+1], total_tokens
        )
        loss.backward()
        accumulated_loss += loss.item()
        print(f"Microbatch {index+1} loss contribution: {loss.item():.6f}")

    torch.testing.assert_close(full_loss.detach(), full_loss.new_tensor(accumulated_loss))
    max_difference = 0.0
    for (name, full_param), (_, micro_param) in zip(
        full_model.named_parameters(), micro_model.named_parameters()
    ):
        torch.testing.assert_close(full_param.grad, micro_param.grad)
        difference = (full_param.grad - micro_param.grad).abs().max().item()
        max_difference = max(max_difference, difference)
        print(f"{name}: gradients match")

    print(f"Full-batch loss: {full_loss.item():.6f}")
    print(f"Accumulated loss: {accumulated_loss:.6f}")
    print(f"Maximum gradient difference: {max_difference:.10f}")
    print("All parameter gradients match within floating-point tolerance.")


if __name__ == "__main__":
    main()
