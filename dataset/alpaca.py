from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import pandas as pd
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import torch
from dataset.prompt_maker import alpaca_prompt_maker
import transformers
from dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
from functools import partial


class AlpacaDataset():
    def __new__(self, max_length=256, use_prompt_loss=False, limit=64,
                prob_data_display=0.1, data_path='/mnt/bn/pxy/data/alpaca_data.json', valid_data_path=None,
                use_large_data=False, val_set_size=0, micro_batch_size=4, tokenizer="bigscience/bloom-560m"):
        class args:
            max_length=256
            use_prompt_loss=False
            prob_data_display=0.1
            data_path='/mnt/bn/pxy/data/alpaca_data.json'
            valid_data_path=None
            use_large_data=False
            val_set_size=None
            micro_batch_size=4
            tokenizer="YeungNLP/bloomz-396m-zh"
            seed=1
            

        prompt_maker=alpaca_prompt_maker.PromptMaker()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        train_data, val_data = get_json_train_valid_data(
            args=args,
            data_file=args.data_path,
            valid_data_file=args.valid_data_path,
            val_set_size=args.val_set_size,
            prompt_fn=partial(generate_and_tokenize_prompt, args=args, tokenizer=tokenizer, prompt_maker=prompt_maker, verbose=False, use_prompt_labels=True),
        )
        return train_data #[:limit]


class AlpacaDataset_(Dataset):
    def __init__(self, root = 'raw/', tokenizer=None, limit=None, embed_layer=None):
        super().__init__()

        path = os.path.join(root, 'alpaca.parquet')

        self.data = []
        self.end_of_text_token = "<|endoftext|>"
        
        df = pd.read_parquet(path)
        records = json.loads(df.to_json(orient = "records"))
        
        NUM=len(records)
        if limit:
            NUM=limit
        class args:
            max_length=256
            use_prompt_loss=False
            prob_data_display=0.1

        for r in records[:NUM]:
            dp=generate_and_tokenize_prompt(r, args=args, prompt_maker=alpaca_prompt_maker.PromptMaker(), tokenizer=tokenizer, padding=True)
            for key in dp.keys(): dp[key]=torch.tensor(dp[key]).unsqueeze(0)
            self.data.append(dp)

        # if tokenizer:
        #     self.tokenizer=tokenizer
        #     tokenizer.pad_token = tokenizer.eos_token
        #     self.data=self.tokenizer(self.data, return_tensors="pt", padding='longest')['input_ids']
        #     # print(self.tokenizer)
        #     self.tokenized=self.data.to(next(embed_layer.parameters()).device)

        # if embed_layer:
        #     self.data=embed_layer(self.data.to(next(embed_layer.parameters()).device))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # if self.tokenizer:
        #     generated = self.tokenizer.encode(self.data[item])
        #     context = torch.tensor([generated])
        # else:
        #     context=self.data[item]
        # return embed, tokens
        if hasattr(self, 'tokenized'):
            return self.data[item], self.tokenized[item]
        else:
            print(self.data[item]['input_ids'].shape)
            return self.data[item]['input_ids'], self.data[item]['labels']
    
if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained("YeungNLP/bloomz-396m-zh")
    # ds=AlpacaDataset(root='/mnt/bn/pxy/yarn/Influence/dataset/raw', tokenizer=tokenizer)
    # print(ds[0])
    # print(len(ds))
    td=AlpacaDataset()
    print(td)
    # print(td[0])
    res=torch.utils.data.Subset(td, [1,2,3])
    print(res)
    exit()
    class args:
        max_length=256
        use_prompt_loss=False
        prob_data_display=0.1
        data_path='/mnt/bn/pxy/data/alpaca_data.json'
        valid_data_path=None
        use_large_data=False
        val_set_size=0
        micro_batch_size=4

    prompt_maker=alpaca_prompt_maker.PromptMaker()
    train_data, val_data = get_json_train_valid_data(
        args=args,
        data_file=args.data_path,
        valid_data_file=args.valid_data_path,
        val_set_size=args.val_set_size,
        prompt_fn=partial(generate_and_tokenize_prompt, args=args, tokenizer=tokenizer, prompt_maker=prompt_maker, use_prompt_labels=True),
    )
    print(type(train_data))

    train_dataloader = DataLoader(
        train_data,
        shuffle=True, 
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        batch_size=args.micro_batch_size, 
        pin_memory=False,
        drop_last=True
    )
    for d in train_dataloader:
        print(d)
        break