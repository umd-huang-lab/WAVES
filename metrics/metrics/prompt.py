import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# Load GPT-2 large model and tokenizer
def load_perplexity_model_and_tokenizer():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ppl_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    return ppl_model, ppl_tokenizer


# Compute perplexity for a single prompt
def compute_prompt_perplexity(prompt, models, stride=512):
    assert isinstance(prompt, str)
    assert isinstance(models, tuple) and len(models) == 2
    ppl_model, ppl_tokenizer = models
    encodings = ppl_tokenizer(prompt, return_tensors="pt")
    max_length = ppl_model.config.n_positions
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(
            next(ppl_model.parameters()).device
        )
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl
