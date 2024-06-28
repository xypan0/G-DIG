import transformers
import torch
from nngeometry.object import PMatKFAC
import nngeometry
from nngeometry import lm_metrics, llama_layercollection
from torch.utils.data import Subset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pickle
from dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
from functools import partial
from dataset.prompt_maker.contrastive_translate_prompt_maker import PromptMaker

# bloom_ignore_lc=['atte', 'lm_head', '0', '1', '2', '3', '21', '22', '23']
bloom_ignore_lc=['atte', 'lm_head', 'dense_4h_to_h']
# llama_ignore_lc=['attn', 'lm_head', 'up', 'gate', '0', '1', '.2.', '3', '4', '9', '21', '22', '23']
# baichuan_ignore_lc=['attn', 'lm_head', 'up', 'gate', '0', '1', '.2.', '3', '4', '9', '21', '22', '23']
baichuan_ignore_lc=['.0.', '.1.', '.2.', '.4.', '.5.', '.7.', '.8.', '.10.', '.11.', '.13.', '.14.', '.16.', '.17.', '.19.', '.20.', '.22.', '.23.', '.25.', '.26.', '.28.', '.29.', '30', 'attn', 'up', 'gate', 'lm_head']
llama_ignore_lc=['.0.', '.1.', '.2.', '.4.', '.5.', '.7.', '.8.', '.10.', '.11.', '.13.', '.14.', '.16.', '.17.', '.19.', '.20.', '.22.', '.23.', '.25.', '.26.', '.28.', '.29.', '30', 'attn', 'up', 'gate', 'lm_head']

# Specify here which layer you do not want to compute the Fisher information matrix
CUR_LC=bloom_ignore_lc

def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-v', '--device')           # device e.g. 'cuda:1'
    parser.add_argument('-idstart', '--indexestart', type=int)     
    parser.add_argument('-idend', '--indexesend', type=int)
    parser.add_argument('-d', '--data_path') 
    parser.add_argument('-o', '--output') 
    parser.add_argument('-m', '--model') 
    parser.add_argument('-t', '--trials', type=int, default=100) 
    parser.add_argument('-k', '--tokenizer', type=str) 

    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    if CUR_LC==baichuan_ignore_lc:
        tokenizer.pad_token = tokenizer.eos_token

    device=torch.device(args.device)
    model.to(device)

    class tmpargs:
        max_length=256
        use_prompt_loss=False
        prob_data_display=0.1
        data_path=args.data_path
        valid_data_path=None
        use_large_data=False
        val_set_size=None
        micro_batch_size=1
        tokenizer=args.tokenizer
        seed=1
    train_data, val_data = get_json_train_valid_data(
        args=tmpargs,
        data_file=tmpargs.data_path,
        valid_data_file=tmpargs.valid_data_path,
        val_set_size=tmpargs.val_set_size,
        prompt_fn=partial(generate_and_tokenize_prompt, args=tmpargs, verbose=False, tokenizer=tokenizer, prompt_maker=PromptMaker(args=tmpargs)),
    )
    trainset=train_data

    ids=list(range(args.indexestart, args.indexesend+1))
    print(len(ids))
    sub_dataset=torch.utils.data.Subset(trainset, ids)

    trainloader = DataLoader(
        sub_dataset,
        shuffle=True, 
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        batch_size=1, 
        pin_memory=False,
        drop_last=True
    )

    lc=llama_layercollection.LLamaLayerCollection.from_model(model, True, CUR_LC)
    print(f'parameter num: {lc.numel()}')
    F_kfac = lm_metrics.FIM(model=model,
                 loader=trainloader,
                 representation=PMatKFAC,
                 n_output=args.trials,
                 variant='empirical_fisher',
                 device=device, layer_collection=lc)
    
    with open(args.output, 'wb') as f:
        pickle.dump(F_kfac, f)

if __name__=='__main__':
    main()
