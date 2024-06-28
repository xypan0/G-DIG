import json
import torch
from transformers import AutoTokenizer
import argparse
import numpy as np
import subprocess
import datetime
import time
from dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
from functools import partial
from dataset.prompt_maker.contrastive_translate_prompt_maker import PromptMaker

def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-n', '--nsubsets', type=int)     
    parser.add_argument('-k', '--kfac')          
    parser.add_argument('-m', '--model', type=str, default='bigscience/bloom-560m')  
    parser.add_argument('-t', '--tokenizer') 
    parser.add_argument('-l', '--limit', type=int, default=-1)  
    parser.add_argument('-lq', '--limit_query', type=int, default=-1)  
    parser.add_argument('-o', '--output', type=str, default='res.jsonl')  
    parser.add_argument('-d', '--data_path')  
    parser.add_argument('-q', '--query_path') 
    parser.add_argument('-bq', '--batch_query', type=int, default=16)  
    parser.add_argument('-lmd', '--lambdaa', type=float, default=0.5)  
    parser.add_argument('--full-score', default=0, type=int)
    parser.add_argument('--ekfac', default=0, type=int)  
    parser.add_argument('--start', default=-1, type=int) 
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--layer', default='b', type=str)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

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
    if args.limit>0:
        sub_dataset=torch.utils.data.Subset(train_data, range(args.limit))
    elif args.start >= 0 and args.end >= 0:
        ids=list(range(args.start, args.end+1))
        print(len(ids))
        sub_dataset=torch.utils.data.Subset(train_data, ids)
    else:
        sub_dataset=train_data

    trainset=sub_dataset

    def split_array(arr, k):
        chunks = np.array_split(arr, k)
        indices = []
        for chunk in chunks:
            start_index = arr.tolist().index(chunk[0])
            end_index = start_index + len(chunk) - 1
            indices.append((start_index, end_index))
        return indices
    indices=split_array(np.arange(len(trainset)), args.nsubsets)

    filenames=[str(datetime.datetime.timestamp(datetime.datetime.now()))+'.json' for i in range(args.nsubsets)]

    st=time.time()
    childlist=[]
    for idx, (fn, subset) in enumerate(zip(filenames, indices)):
        child = subprocess.Popen(['python3', 'query_loss_mapper.py', '-v', f'cuda:{idx}', \
                                    '-idstart', f'{subset[0]}', '-idend', f'{subset[1]}', \
                                    '-o', f'{fn}', '-m', f'{args.model}', '-l', f'{args.limit}', \
                                    '-t', f'{args.tokenizer}', '-d', f'{args.data_path}', \
                                    '-bq', f'{args.batch_query}', '-q', f'{args.query_path}', \
                                    '-k', f'{args.kfac}', '-lq', f'{args.limit_query}', '-lmd', f'{args.lambdaa}', \
                                    '--full-score', f'{args.full_score}', '--ekfac', f'{args.ekfac}', \
                                    '--layer', f'{args.layer}'])
        childlist.append(child)

    for child in childlist:
        print(f'Mapper process {child.pid} is running')

    while True:
        flag=True
        print('Checking status...')
        for child in childlist:
            if child.poll() is None:
                flag=False
            else:
                print(f'Mapper process {child.pid} finished')
        if flag:
            break
        time.sleep(5)

    elapse=time.time()-st
    print(f'Mapper processes all finished. Use {elapse}s.')
    print('Reducing...')
    
    
    json_list=[]
    for fn in filenames:
        json_list.extend(json.loads(open(fn).read()))
    
    res_json=sorted(json_list, key=lambda x: x['score'])
    open(args.output, 'w').write(json.dumps(res_json, indent=4, ensure_ascii=False))

    print('Finish reduce')

    
if __name__=='__main__':
    main()