import torch
from transformers import AutoTokenizer
import argparse
import pickle
import numpy as np
import subprocess
import datetime
import time
from dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
from functools import partial
from dataset.prompt_maker.contrastive_translate_prompt_maker import PromptMaker

def to_device(kfac, device='cpu'):
    for key in kfac.data.keys():
        kfac.data[key]=(kfac.data[key][0].to(device), kfac.data[key][1].to(device))
    return kfac

def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-g', '--device')           # device e.g. 'cuda:1'
    parser.add_argument('-n', '--nsubsets', type=int)      # option that takes a value
    parser.add_argument('-d', '--data_path')  # on/off flag
    parser.add_argument('-o', '--output')  # on/off flag
    parser.add_argument('-m', '--model', type=str, default='bigscience/bloom-560m')  # on/off flag
    parser.add_argument('-t', '--trials', type=int, default=1)  # on/off flag
    parser.add_argument('-k', '--tokenizer', type=str)  # on/off flag

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
    trainset=train_data

    def split_array(arr, k):
        chunks = np.array_split(arr, k)
        indices = []
        for chunk in chunks:
            start_index = arr.tolist().index(chunk[0])
            end_index = start_index + len(chunk) - 1
            indices.append((start_index, end_index))
        return indices
    indices=split_array(np.arange(len(trainset)), args.nsubsets)

    filenames=[str(datetime.datetime.timestamp(datetime.datetime.now()))+'.kfac' for i in range(args.nsubsets)]

    st=time.time()
    childlist=[]
    for idx, (fn, subset) in enumerate(zip(filenames, indices)):
        child = subprocess.Popen(['python3', 'kfac_mapper.py', '-v', f'cuda:{idx}', \
                                    '-idstart', f'{subset[0]}', '-idend', f'{subset[1]}', \
                                    '-o', f'{fn}', '-m', f'{args.model}', \
                                    '-t', f'{args.trials}', '-d', f'{args.data_path}', '-k', f'{args.tokenizer}'])
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
    

    device=torch.device(args.device)
    
    kfac_list=[]
    for fn in filenames:
        with open(fn, 'rb') as f:
            pF = pickle.load(f)
            kfac_list.append(pF)
    reduce=kfac_list[0]
    for kfac in kfac_list[1:]:
        for key in kfac.data.keys():
            reduce.data[key][0].add_(kfac.data[key][0].to(device))
            reduce.data[key][1].add_(kfac.data[key][1].to(device))

    for key in reduce.data.keys():
        reduce.data[key][0].div_(args.nsubsets)
        reduce.data[key][1].div_(args.nsubsets)

    reduce=to_device(reduce, 'cpu')
    with open(args.output, 'wb') as f:
        pickle.dump(reduce, f)

    print('Finish reduce')

    
if __name__=='__main__':
    main()