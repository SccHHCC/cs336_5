"""Compare full-batch gradients with equal-size microbatch accumulation."""

import torch
import torch.nn.functional as F


def make_model():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.5)
    return model


def run(mode, x, y):
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    if mode == "full_batch":
        F.mse_loss(model(x), y).backward()
    else:
        steps = 8
        for small_x, small_y in zip(x.chunk(steps), y.chunk(steps)):
            if mode == "clear_each_time":
                optimizer.zero_grad()
            loss = F.mse_loss(model(small_x), small_y)
            (loss / steps).backward()
    gradient = model.weight.grad.item()
    optimizer.step()
    return gradient, model.weight.item()


def main():
    # A tiny regression model isolates accumulation; this is not an LM/SFT test.
    x = torch.arange(1, 33, dtype=torch.float64).reshape(-1, 1) / 32
    y = 2 * x
    results = {}
    for mode in ["full_batch", "accumulate", "clear_each_time"]:
        results[mode] = run(mode, x, y)
        gradient, weight = results[mode]
        print(f"{mode}: gradient={gradient:.9f}, updated_weight={weight:.9f}")
    torch.testing.assert_close(
        torch.tensor(results["full_batch"], dtype=torch.float64),
        torch.tensor(results["accumulate"], dtype=torch.float64),
    )
    print("Full batch and accumulation agree.")


if __name__ == "__main__":
    main()
