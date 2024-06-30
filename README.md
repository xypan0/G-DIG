# G-DIG: Towards Gradient-based DIverse and hiGh-quality Instruction Data Selection for Machine Translations

In this implementation, we use model bigscience/bloom-560m for demonstration. And we use a demo dataset demo.json for demonstration. There are three steps in our proposed high-quality selection method in G-DIG:

1. Fine-tune the target model with candidate data $\mathcal{D}_{raw}$ using huggingface compatible model.

```
Save checkpoint to ./checkpoint, which will be used for calculate the Hessian matrix and scoring.
```

2. Compute the Hessian matrix.

```
./hessian.sh

# In this script, demo.json should be replaced by the training data you used to finetune the LLM.
```

3. run influence function to compute data score.

``` 
./if_score.sh

# In this script, -d demo.json corresponds to the candidate dataset and -q demo.json corresponds to the seed dataset.
```

Finally, use the data score according to Equation (4) in the paper to select high-quality data.


## Data

We release our selected data (EN->ZH and DE->EN)

1. DE->EN training data of size from 1k to 64k available [here](https://drive.google.com/file/d/1aklM0Q7BV14tVZF8isQdqe_PbpEwHEv9/view)
2. ZH->EN training data to be continue.

## Citation

If this repo was useful to you, please consider citing

```
@article{pan2024g,
  title={G-DIG: Towards Gradient-based DIverse and hiGh-quality Instruction Data Selection for Machine Translation},
  author={Pan, Xingyuan and Huang, Luyang and Kang, Liyan and Liu, Zhicheng and Lu, Yu and Cheng, Shanbo},
  journal={arXiv preprint arXiv:2405.12915},
  year={2024}
}
```
