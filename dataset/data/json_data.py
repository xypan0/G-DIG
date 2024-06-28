
from dataset.data.dataset import (
    DynamicPromptDataset,
    StreamDynamicPromptDataset,
    COAIDynamicPromptDataset,
    COAIStreamDynamicPromptDataset
)
import json
import random
from dataset.utils.io_utils import load_json, grob_paths
import random
import os
import logging
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_json_train_valid_data(
    args,
    val_set_size: int, 
    data_file: str = "alpaca_data_cleaned.json", 
    train_val_provider = None,
    prompt_fn = None,
    valid_data_file: str = None,
    for_coai: bool = False,
):
    assert prompt_fn is not None, "please provide prompt_fn"

    if train_val_provider is not None:
        train_val_provider(data_file, valid_data_file, val_set_size)

    if args.use_large_data:
        assert valid_data_file is not None, \
            """
            For large data, training data is iterable among large mounts of files,
            It is not convinient to fetch data as validation set that won't be 
            resampled again as training data (from an iterable pipeline).
            So it is recommended to provide validation data yourself.
            """
        train_dataloader_class = COAIStreamDynamicPromptDataset if for_coai else StreamDynamicPromptDataset
    else:
        train_dataloader_class = COAIDynamicPromptDataset if for_coai else DynamicPromptDataset
    valid_dataloader_class = COAIDynamicPromptDataset if for_coai else DynamicPromptDataset


    # if validation data is provided, just use it.
    if valid_data_file:
        train_data = train_dataloader_class(
            args=args,
            json_data=data_file, 
            dynamic_transform=prompt_fn, 
            shuffle=False, 
            from_file=True)

        valid_data = valid_dataloader_class(
            args=args,
            json_data=valid_data_file, 
            static_transform=prompt_fn, 
            from_file=True,
            shuffle=False)

    # if validation data is not provided, produce pseudo validation data from training data.
    else:
        if val_set_size:
            raw_train_data = load_json(grob_paths(data_file))
            random.seed(args.seed)
            random.shuffle(raw_train_data)

            train_data = train_dataloader_class(
                args=args,
                json_data=raw_train_data[:-val_set_size], 
                dynamic_transform=prompt_fn, 
                shuffle=True, 
                from_file=False)

            valid_data = valid_dataloader_class(
                args=args,
                json_data=raw_train_data[-val_set_size:], 
                static_transform=prompt_fn, 
                from_file=False,
                shuffle=False)
        else:
            train_data = train_dataloader_class(
                args=args,
                json_data=data_file, 
                dynamic_transform=prompt_fn, 
                shuffle=False, 
                from_file=True)
            valid_data = None

    return train_data, valid_data


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
        # user_prompt=user_prompt
        pass
    # --------
    len_user_prompt_tokens = len(user_prompt) 
    len_user_prompt_tokens = min(len_user_prompt_tokens, args.max_length)

    full_tokens = tokenizer(
        full_text,
        truncation=truncation,
        max_length=args.max_length # 
    )["input_ids"]
    # ---------
    if full_tokens[-1] != tokenizer.eos_token_id: 
        full_tokens=full_tokens+[tokenizer.eos_token_id]
    else:
        full_tokens=full_tokens
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


def generate_and_tokenize_prompt_with_contrastive_label(
    data_point,
    args = None,
    tokenizer = None,
    prompt_maker = None,
    use_prompt_labels = True,
    padding: bool = False,
    truncation: bool = True,
    verbose: bool = True,
    ignore_loss_idx: int = -100,
    path_contrastive_label = None,
):
    assert prompt_maker is not None, "please provide prompt_maker"
    assert path_contrastive_label is not None
    input_text = prompt_maker.get_input(data_point)
    target_text = prompt_maker.get_target(data_point)
    full_text = input_text + target_text
    # print(prompt_maker)
    c_target_text = prompt_maker.get_constrastive_target(data_point, path_contrastive_label)
    c_full_text = input_text + c_target_text

    user_prompt = tokenizer(
        input_text,
        truncation=True,
        max_length=args.max_length + 1,
    )["input_ids"][:-1] # no eos token

    # --------
    # c_target_text_t = tokenizer(
    #     c_target_text,
    #     truncation=True,
    #     max_length=args.max_length + 1,
    # )["input_ids"]
    # # print(c_target_text, c_target_text_t)
    # if not c_target_text_t: c_target_text_t=[tokenizer.eos_token_id]
    # if c_target_text_t[-1] != tokenizer.eos_token_id: 
    #     c_target_text_t=c_target_text_t+[tokenizer.eos_token_id]
    # else:
    #     # c_target_text_t=c_target_text_t
    #     pass
    # --------
    user_prompt = tokenizer(
        input_text,
        truncation=True,
        max_length=args.max_length + 1,
    )["input_ids"]
    if user_prompt[-1]==tokenizer.eos_token_id:
        user_prompt=user_prompt[:-1]
    else:
        # user_prompt=user_prompt
        pass    
    #---------
    len_user_prompt_tokens = len(user_prompt) 
    # len_c_target_text_tokens = len(c_target_text_t) 
    len_user_prompt_tokens = min(len_user_prompt_tokens, args.max_length)

    full_tokens = tokenizer(
        full_text,
        truncation=truncation,
        max_length=args.max_length # 
    )["input_ids"]
    # ---------
    if full_tokens[-1] != tokenizer.eos_token_id: 
        full_tokens=full_tokens+[tokenizer.eos_token_id]
    else:
        full_tokens=full_tokens
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

    # if verbose and (random.random() <= args.prob_data_display):
    #     logger.info(f"""### random data case:
    #      batch length       = {len(full_tokens)}
    # (P)  prompt             = {[input_text]}
    # (PT) prompt_and_target  = {[full_text]}
    # (PT) tokenized          = {full_tokens}
    # (PT) attention_mask     = {attention_mask}
    # (PT) labels             = {labels}
    # """)
        

    #-----------deal with c_labels-------

    c_full_tokens = tokenizer(
        c_full_text,
        truncation=truncation,
        max_length=args.max_length # 
    )["input_ids"]
    if len(c_full_tokens)==args.max_length:
        pass
    else:
        if c_full_tokens[-1] != tokenizer.eos_token_id: 
            c_full_tokens=c_full_tokens+[tokenizer.eos_token_id]
        else:
            pass
    # ---------
    # if c_full_tokens[-1] != tokenizer.eos_token_id: 
    #     c_full_tokens=c_full_tokens+[tokenizer.eos_token_id]
    # else:
    #     c_full_tokens=c_full_tokens
    # ---------
    # attention_mask = [1] * len(c_full_tokens)

    if args.use_prompt_loss:
        c_labels = copy.deepcopy(c_full_tokens)
    else:
        # c_labels = [ignore_loss_idx] * (len(full_tokens)-len_c_target_text_tokens) + c_full_tokens[len_user_prompt_tokens:]
        c_labels = [ignore_loss_idx] * len_user_prompt_tokens + c_full_tokens[len_user_prompt_tokens:]

    ## deal with padding
    if padding:
        padded_length = args.max_length - len(c_full_tokens)
        c_full_tokens.extend([tokenizer.pad_token_id] * padded_length)
        c_labels.extend([ignore_loss_idx] * padded_length)
        # attention_mask = attention_mask + [0] * padded_length

    if verbose and (random.random() <= args.prob_data_display):
        logger.info(f"""### random data case:
         batch length       = {len(full_tokens)}
    (P)  prompt             = {[input_text]}
    (PT) prompt_and_target  = {[full_text]}
    (PT) cons_prompt_and_target  = {[c_full_text]}
    (PT) tokenized          = {full_tokens}
    (PT) attention_mask     = {attention_mask}
    (PT) labels             = {labels}
    (PT) contrastive labels = {c_labels}
    """)

    #------------------------------------

    ## deal with prompt or not (w.r.t. pretrain and instruction tuning)
    if use_prompt_labels:
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        return {
            "input_ids": full_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
            "c_labels": c_labels,
        }
    else:
        return {
            "input_ids": full_tokens,
            "attention_mask": attention_mask,
        }

if __name__ == '__main__':
    from ..utils.prompt_maker.custum_prompt_maker import PromptMaker
    from transformers import LlamaTokenizer
    from functools import partial

    class temp_args():
        max_length = 256
        model_id = "/mnt/bn/multilingual-translation/public/hf_models/llama-7b-hf/"
        data_file = "/opt/tiger/llama/finetune/alpaca-lora/codes/data/alpaca_data_cleaned.json"

    args = temp_args()
    tokenizer = LlamaTokenizer.from_pretrained(args.model_id, add_eos_token=True)
    tokenizer.pad_token_id = 0 # unk. we want this to be different from the eos token


    train_data = DynamicPromptDataset(
        json_data=args.data_file, 
        dynamic_transform=partial(generate_and_tokenize_prompt, args=args, tokenizer=tokenizer, prompt_maker=PromptMaker()), 
        shuffle=True, 
        from_file=True
    )

    for data in train_data:
        print(data)
