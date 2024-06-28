# -g: GPU device for reduce
# -n: Number of GPUs (Parallelism)
# -d: Dataset
# -m: Model
# -k: Tokenizer
# -o: Output file

python3 kfac_launcher.py \
    -g cuda:0 \
    -n 8 \
    -d demo.json \
    -m ./checkpoint \
    -t 1 \
    -k bigscience/bloom-560m \
    -o hessian.pkl \

