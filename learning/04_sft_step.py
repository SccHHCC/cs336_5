"""Exercise existing SFT helpers with a tiny CPU model, without importing vLLM."""

import ast
import __future__
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def load_helpers():
    # Load the actual function bodies without unrelated module-level imports.
    path = Path(__file__).resolve().parents[1] / "cs336_alignment" / "sft_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "compute_entropy", "get_response_log_probs",
        "masked_normalize", "sft_microbatch_train_step",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch, "F": F}
    code = compile(
        ast.Module(body=functions, type_ignores=[]), str(path), "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)
    return namespace


class TinyModel(torch.nn.Module):
    """A teaching model: predicts from the current token, not full context."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def main():
    torch.manual_seed(42)
    helpers = load_helpers()
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Prompt = [1, 2]; response A = [3, 4, 5], response B = [6]. Pad = 0.
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 6, 0]])
    labels = torch.tensor([[2, 3, 4, 5], [2, 6, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 0]])
    constant = 2.0
    optimizer.zero_grad()
    outputs = helpers["get_response_log_probs"](
        model, input_ids, labels, return_token_entropy=True,
    )
    log_probs = outputs["log_probs"]
    log_probs.retain_grad()
    loss, metadata = helpers["sft_microbatch_train_step"](
        log_probs, mask, gradient_accumulation_steps=1,
        normalize_constant=constant,
    )
    # Independent cross-entropy calculation checks target selection and scaling.
    with torch.no_grad():
        logits = model(input_ids).logits
        token_losses = F.cross_entropy(
            logits.reshape(-1, 8), labels.reshape(-1), reduction="none",
        ).reshape_as(labels)
        expected = (token_losses * mask).sum() / constant / labels.shape[0]
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(log_probs.grad, -mask.float() / constant / 2)
    print("input shape:", tuple(input_ids.shape))
    print("target log-probs:\n", log_probs.detach())
    print("entropy shape:", tuple(outputs["token_entropy"].shape))
    print("loss before update:", loss.item())
    print("target log-prob gradients:\n", log_probs.grad)
    before = model.head.weight.detach().clone()
    optimizer.step()
    print("parameters changed:", not torch.equal(before, model.head.weight))
    with torch.no_grad():
        new_log_probs = helpers["get_response_log_probs"](
            model, input_ids, labels,
        )["log_probs"]
        new_loss = (-new_log_probs * mask).sum() / constant / 2
    print("loss after update:", new_loss.item())
    print("Target selection, normalization, and masked gradients checked.")


if __name__ == "__main__":
    main()
