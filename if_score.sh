# -k: Hessian file
# -n: Number of GPUs (Parallelism)
# -m: Model
# -t: Tokenizer
# -o: Output file
# -d: Candidate dataset
# -q: Seed dataset
# -bq: Batch size of seed dataset
# -lmd: Lambda in Hessian inverse



python3 query_loss_launcher.py \
    -n 8 -k hessian.pkl \
    -m ./checkpoint \
    -t bigscience/bloom-560m \
    -o score_results.json \
    -d demo.json \
    -q demo.json \
    -bq 2 -lmd 0.5 --full-score 1