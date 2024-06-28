import torch
from nngeometry.object import PMatEKFAC
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pickle
import json
from nngeometry.object import lm_vector
import transformers
import time
from dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
from functools import partial
from dataset.prompt_maker.translate_prompt_maker import PromptMaker
from tqdm import tqdm
from nngeometry.maths import kronecker

def to_device(kfac, device):
    for key in kfac.data.keys():
        kfac.data[key]=(kfac.data[key][0].to(device), kfac.data[key][1].to(device))
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return kfac

def to_ekfac_and_device(kfac, device):
    for key in kfac.data.keys():
        kfac.data[key]=(kfac.data[key][0].to(device), kfac.data[key][1].to(device))
    # ---------
    evecs = dict()
    diags = dict()

    kfac_blocks = kfac.data
    for layer_id, layer in \
            kfac.generator.layer_collection.layers.items():
        a, g = kfac_blocks[layer_id]
        evals_a, evecs_a = torch.linalg.eigh(a)
        evals_g, evecs_g = torch.linalg.eigh(g)
        evecs[layer_id] = (evecs_a, evecs_g)
        diags[layer_id] = kronecker(evals_g.view(-1, 1),
                                    evals_a.view(-1, 1))
        del a, g, kfac_blocks[layer_id]
    data = (evecs, diags)
    ekfac=PMatEKFAC(generator=kfac.generator, data=data)
    return ekfac

# bloom_ignored_layers=['atte', 'lm_head', '0', '1', '2', '3', '21', '22', '23']
bloom_ignore_lc=['atte', 'lm_head', 'dense_4h_to_h']
# baichuan_ignore_lc=['attn', 'lm_head', 'up', 'gate', '0', '1', '.2.', '3', '4', '9', '21', '22', '23']
baichuan_ignore_lc=['.0.', '.1.', '.2.', '.4.', '.5.', '.7.', '.8.', '.10.', '.11.', '.13.', '.14.', '.16.', '.17.', '.19.', '.20.', '.22.', '.23.', '.25.', '.26.', '.28.', '.29.', '30', 'attn', 'up', 'gate', 'lm_head']
llama_ignore_lc=['.0.', '.1.', '.2.', '.4.', '.5.', '.7.', '.8.', '.10.', '.11.', '.13.', '.14.', '.16.', '.17.', '.19.', '.20.', '.22.', '.23.', '.25.', '.26.', '.28.', '.29.', '30', 'attn', 'up', 'gate', 'lm_head']
# llama_ignore_lc=['attn', 'lm_head', 'up', 'gate', '0', '1', '.2.', '3', '4', '9', '21', '22', '23']
CUR_LC=bloom_ignore_lc

def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-k', '--kfac')           # kfac_dir
    parser.add_argument('-m', '--model')  
    parser.add_argument('-t', '--tokenizer')  
    parser.add_argument('-v', '--device')           # device e.g. 'cuda:1'
    parser.add_argument('-l', '--limit', type=int, default=-1)  
    parser.add_argument('-lq', '--limit_query', type=int, default=-1)  
    parser.add_argument('-o', '--output', type=str, default='res.jsonl')  
    parser.add_argument('-d', '--data_path')  
    parser.add_argument('-q', '--query_path')  
    parser.add_argument('-bq', '--batch_query', type=int, default=16)  
    parser.add_argument('-idstart', '--indexestart', type=int)    
    parser.add_argument('-idend', '--indexesend', type=int)
    parser.add_argument('-lmd', '--lambdaa', type=float, default=0.5)  

    parser.add_argument('--full-score', default=0, type=int) 
    parser.add_argument('--ekfac', default=0, type=int) 
    parser.add_argument('--layer', default='b', type=str) 


    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    

    model.to(args.device)
    model.zero_grad()
    with open(args.kfac, 'rb') as f:

        pF = pickle.load(f)
        if args.ekfac:
            pF=to_ekfac_and_device(pF, args.device)
        else:
            pF=to_device(pF, args.device)

    class tmpQueryargs:
        max_length=256
        use_prompt_loss=False
        prob_data_display=0.1
        data_path=args.query_path
        valid_data_path=None
        use_large_data=False
        val_set_size=None
        micro_batch_size=4
        tokenizer=args.tokenizer
        seed=1
    query_data, val_data = get_json_train_valid_data(
        args=tmpQueryargs,
        data_file=tmpQueryargs.data_path,
        valid_data_file=tmpQueryargs.valid_data_path,
        val_set_size=tmpQueryargs.val_set_size,
        prompt_fn=partial(generate_and_tokenize_prompt, args=tmpQueryargs, verbose=False, tokenizer=tokenizer, prompt_maker=PromptMaker(args=tmpQueryargs)),
    )

    if args.limit_query>0:
        query_data=torch.utils.data.Subset(query_data, range(args.limit_query))
    else:
        pass
    queryloader = DataLoader(
        query_data,
        shuffle=False, 
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        batch_size=args.batch_query, 
        pin_memory=False,
        drop_last=True
    )

    d_ihvp=[]
    for q in tqdm(queryloader):
        model.zero_grad()
        inp=q['input_ids'].to(args.device)
        labels=q['labels'].to(args.device)
        
        loss=model(input_ids=inp, labels=labels).loss
        loss.backward()

        vec_query=lm_vector.PVector.from_model_grad(model, ignore_layers=CUR_LC)


        ihvp=pF.inverse(regul=args.lambdaa).mv(vec_query)
        ihvp=lm_vector.PVector(layer_collection=ihvp.layer_collection, 
                           vector_repr=ihvp.vector_repr, dict_repr=ihvp.dict_repr)
        ihvp.svd()
        d_ihvp.append(ihvp)

    
    class tmpargs:
        max_length=256
        use_prompt_loss=False
        prob_data_display=0.1
        data_path=args.data_path
        valid_data_path=None
        use_large_data=False
        val_set_size=None
        micro_batch_size=4
        tokenizer=args.tokenizer
        seed=1
    train_data, val_data = get_json_train_valid_data(
        args=tmpargs,
        data_file=tmpargs.data_path,
        valid_data_file=tmpargs.valid_data_path,
        val_set_size=tmpargs.val_set_size,
        prompt_fn=partial(generate_and_tokenize_prompt, args=tmpargs, verbose=False, tokenizer=tokenizer, prompt_maker=PromptMaker(args=tmpargs)),
    )
    candidate_set=train_data 

    ids=list(range(args.indexestart, args.indexesend+1))
    print(len(ids))
    sub_dataset=torch.utils.data.Subset(candidate_set, ids)

    trainloader = DataLoader(
        sub_dataset,
        shuffle=False, 
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        batch_size=1, 
        pin_memory=False,
        drop_last=True
    )


    st=time.time()
    res_json=[]
    for data in tqdm(trainloader):
        inp=data['input_ids'].to(args.device)
        labels=data['labels'].to(args.device)

        model.zero_grad()
        loss=model(input_ids=inp, labels=labels).loss
        loss.backward()

        vec_candi=lm_vector.PVector.from_model_grad(model, CUR_LC)
        vec_candi.svd()
        
        score=0
        score_list=[]

        for ihvp in d_ihvp:
            tmp=-(vec_candi.dot_svd(ihvp))
            score=score+tmp
            score_list.append(tmp.item())


        score=score/len(d_ihvp)
        text=tokenizer.batch_decode(data['input_ids'], skip_special_tokens=True)
        if args.full_score:
            res_json.append({'score': score.item(), 'score_list': score_list, 'loss': loss.item(), 'text': text})
        else:
            res_json.append({'score': score.item(), 'loss': loss.item(), 'text': text})


    el=time.time()-st
    print(f'Elapse: {el}s.')
    
    res_json=sorted(res_json, key=lambda x: x['score'])

    open(args.output, 'w').write(json.dumps(res_json, indent=4, ensure_ascii=False))

        
if __name__=='__main__':
    main()
