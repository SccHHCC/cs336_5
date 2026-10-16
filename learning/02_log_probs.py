"""CPU lesson: select target log-probabilities and inspect gradients."""

import torch


def main():
    # One sample, two prediction positions, three vocabulary entries.
    initial_probs = torch.tensor([[[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]]])
    logits = initial_probs.log().detach().requires_grad_(True)
    labels = torch.tensor([[1, 2]])
    response_mask = torch.tensor([[1, 0]])

    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    loss = (-target_log_probs * response_mask).sum() / response_mask.sum()

    print("probabilities:\n", logits.softmax(dim=-1).detach())
    print("model argmax:", logits.argmax(dim=-1))
    print("labels:", labels)
    print("target log-probs:", target_log_probs.detach())
    print("response mask:", response_mask)
    print("loss:", loss.item())

    loss.backward()
    print("logit gradients:\n", logits.grad)
    # This updates these toy logits directly, not a language model's weights.
    with torch.no_grad():
        updated_logits = logits - 0.1 * logits.grad
    print("probabilities after one toy update:\n", updated_logits.softmax(dim=-1))


if __name__ == "__main__":
    main()
