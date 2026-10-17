"""Day 1: implement the SFT primitives yourself before viewing existing code."""

import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs=torch.log_softmax(logits, dim=-1)
    probs=log_probs.exp()
    result=-torch.sum(probs*log_probs,dim=-1)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    tensor_masked=tensor*mask
    masked_sum=tensor_masked.sum(dim=dim)
    result=masked_sum/normalize_constant
    return result


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    response_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": response_log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
    return result


def sft_microbatch_train_step(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    normalize_constant=1.0,
):
    normalized_sum = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
    )
    batch_size = policy_log_probs.shape[0]
    loss = normalized_sum / (
        batch_size * gradient_accumulation_steps
    )
    loss.backward()

    metadata = {"loss": loss.detach()}
    return loss, metadata

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    all_ids = []
    all_masks = []

    for prompt, response in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False
        )
        response_ids = tokenizer.encode(
            response, add_special_tokens=False
        )

        combined_ids = prompt_ids + response_ids
        full_mask = (
            [0] * len(prompt_ids)
            + [1] * len(response_ids)
        )

        all_ids.append(combined_ids)
        all_masks.append(full_mask)

    max_length = max(len(ids) for ids in all_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_input_ids = []
    all_labels = []
    all_response_masks = []

    for ids, mask in zip(all_ids, all_masks):
        pad_length = max_length - len(ids)

        padded_ids = ids + [pad_id] * pad_length
        padded_mask = mask + [0] * pad_length

        all_input_ids.append(padded_ids[:-1])
        all_labels.append(padded_ids[1:])
        all_response_masks.append(padded_mask[1:])

    return {
    "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
    "labels": torch.tensor(all_labels, dtype=torch.long),
    "response_mask": torch.tensor(
        all_response_masks, dtype=torch.long
    ),
    }

if __name__ == "__main__":
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]])
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])
    print("entropy:", compute_entropy(logits))
    print("normalized by row:", masked_normalize(values, mask, 2.0, dim=1))
    print("normalized total:", masked_normalize(values, mask, 2.0))
