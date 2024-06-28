import logging

deepspeed_filedict = {
    "ds_fp16_zero3_offload": "/opt/tiger/llama/finetune/alpaca-lora/codes/config/ds_fp16_zero3_offload.json",
    "ds_bf16_zero3_offload": "/opt/tiger/llama/finetune/alpaca-lora/codes/config/ds_bf16_zero3_offload.json",
    "ds_int8_zero3_offload": "/opt/tiger/llama/finetune/alpaca-lora/codes/config/ds_int8_zero3_offload.json",
    "ds_fp16_zero2": "/opt/tiger/llama/finetune/alpaca-lora/codes/config/ds_fp16_zero2.json",
    "ds_int8_zero2": "/opt/tiger/llama/finetune/alpaca-lora/codes/config/ds_int8_zero2.json",
}


def get_deepspeed_config_or_file(deepspeed_file_or_dict: str):
    deepspeed_config = deepspeed_filedict.get(deepspeed_file_or_dict, deepspeed_file_or_dict)
    logging.info(f"load deepspeed file from {deepspeed_config}")
    return deepspeed_config