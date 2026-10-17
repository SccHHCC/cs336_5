"""Observe one CPU update using the learner's SFT functions."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


spec = importlib.util.spec_from_file_location(
    "practice", Path(__file__).with_name("05_sft_practice.py")
)
practice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(practice)


class TinyModel(torch.nn.Module):
    # Teaching model only: each prediction sees the current token, not full context.
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def main():
    torch.manual_seed(42)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 6, 0]])
    labels = torch.tensor([[2, 3, 4, 5], [2, 6, 0, 0]])
    response_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 0]])
    total_tokens = response_mask.sum()

    optimizer.zero_grad()
    before = [p.detach().clone() for p in model.parameters()]
    outputs = practice.get_response_log_probs(
        model, input_ids, labels, return_token_entropy=True
    )
    log_probs = outputs["log_probs"]
    log_probs.retain_grad()
    loss = practice.masked_normalize(-log_probs, response_mask, total_tokens)
    reference = F.cross_entropy(
        model(input_ids).logits.reshape(-1, 8), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    torch.testing.assert_close(loss, (reference * response_mask).sum() / total_tokens)
    assert outputs["token_entropy"].shape == labels.shape
    print("Effective response tokens:", total_tokens.item())
    print("Loss before update:", round(loss.item(), 6))

    loss.backward()
    torch.testing.assert_close(log_probs.grad, -response_mask.float() / total_tokens)
    unchanged = all(torch.equal(old, p) for old, p in zip(before, model.parameters()))
    assert unchanged
    print("Parameters unchanged after backward:", unchanged)
    print("Gradient on selected log-probs:\n", log_probs.grad)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print("Parameter gradient norm before clipping:", round(norm.item(), 6))
    optimizer.step()
    changed = any(not torch.equal(old, p) for old, p in zip(before, model.parameters()))
    assert changed
    print("Parameters changed after step:", changed)
    with torch.no_grad():
        updated = practice.get_response_log_probs(model, input_ids, labels)["log_probs"]
        new_loss = practice.masked_normalize(-updated, response_mask, total_tokens)
    print("Loss after update:", round(new_loss.item(), 6))
    print("Target selection, masked loss, and gradient checks passed.")


if __name__ == "__main__":
    main()
