import random
import logging
import copy

logger=logging.getLogger()


def generate_and_tokenize_prompt(
    data_point,
    args = None,
    tokenizer = None,
    prompt_maker = None,
    use_prompt_labels = True,
    padding: bool = False,
    truncation: bool = True,
    verbose: bool = True,
    ignore_loss_idx: int = -100,
):
    assert prompt_maker is not None, "please provide prompt_maker"
    input_text = prompt_maker.get_input(data_point)
    target_text = prompt_maker.get_target(data_point)
    full_text = input_text + target_text
    
    user_prompt = tokenizer(
        input_text,
        truncation=True,
        max_length=args.max_length + 1,
    )["input_ids"][:-1] # no eos token

    # --------
    user_prompt = tokenizer(
        input_text,
        truncation=True,
        max_length=args.max_length + 1,
    )["input_ids"]
    if user_prompt[-1]==tokenizer.eos_token_id:
        user_prompt=user_prompt[:-1]
    else:
        user_prompt=user_prompt[1:]
    # --------
    len_user_prompt_tokens = len(user_prompt) 
    len_user_prompt_tokens = min(len_user_prompt_tokens, args.max_length)

    full_tokens = tokenizer(
        full_text,
        truncation=truncation,
        max_length=args.max_length
    )["input_ids"]
    # ---------
    if full_tokens[-1] != tokenizer.eos_token_id: 
        full_tokens=full_tokens+[tokenizer.eos_token_id]
    # ---------
    attention_mask = [1] * len(full_tokens)

    if args.use_prompt_loss:
        labels = copy.deepcopy(full_tokens)
    else:
        labels = [ignore_loss_idx] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:]

    ## deal with padding
    if padding:
        padded_length = args.max_length - len(full_tokens)
        full_tokens.extend([tokenizer.pad_token_id] * padded_length)
        labels.extend([ignore_loss_idx] * padded_length)
        attention_mask = attention_mask + [0] * padded_length

    if verbose and (random.random() <= args.prob_data_display):
        logger.info(f"""### random data case:
         batch length       = {len(full_tokens)}
    (P)  prompt             = {[input_text]}
    (PT) prompt_and_target  = {[full_text]}
    (PT) tokenized          = {full_tokens}
    (PT) attention_mask     = {attention_mask}
    (PT) labels             = {labels}
    """)
    # print(len(full_tokens))
    ## deal with prompt or not (w.r.t. pretrain and instruction tuning)
    if use_prompt_labels:
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        return {
            "input_ids": full_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    else:
        return {
            "input_ids": full_tokens,
            "attention_mask": attention_mask,
        }