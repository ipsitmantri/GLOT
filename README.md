# Towards Improved Sentence Representations Using Token Graphs


This repository is the official implementation of [Towards Improved Sentence Representations Using Token Graphs](https://openreview.net/forum?id=stMX9KBhUI). 

<image src="glot.png" width="100%">


## Environment Setup

Clone the repository:
```bash
git clone git@github.com:ipsitmantri/GLOT.git
cd GLOT
```

Create a virtual environment in python (recommended to use Python 3.10):

```bash
conda create -n glot python=3.10 -y
conda activate glot
```

Install Dependencies
```bash
pip install -r requirements.txt
```

## HuggingFace Setup &#x1F917;
Run the command:
```bash
>>> hf auth login
```
and follow the instructions to setup the [User Access Token](https://huggingface.co/docs/hub/security-tokens). Additionally, copy the token to `HF_TOKEN="<>" # Place your huggingface token here` in `main.py` and `diagnostic_stress_test.py`.

Next, download the [MS-MARCO Triplets](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz) dataset and extract it to `./data/msmarco-triplets.jsonl`


## Evaluation on GLUE Benchmark

```bash
python main.py \
  --model_name_or_path='bert-base-uncased' \      # Any model from huggingface
  --decoder_cls_last_token=0 \    # Other choices: [1]
  --task=cola \     # Other choices: [sst2, stsb, mrpc, qqp, mnli, qnli, rte, wnli]
  --max_length=128 \
  --adaptive_length=0 \
  --epochs=3 \
  --batch_size=32 \
  --eval_batch_size=64 \
  --lr=2e-4 \
  --weight_decay=0.0 \
  --seed=42 \
  --verbose=1 \
  --pooling_method=glot \      # Other choices: [cls, mean, max, adapool]
  --gnn_type=gat \
  --scorer_hidden=128 \
  --gat_hidden_dim=256 \
  --num_layers=2 \
  --jk_mode=cat \
  --graph_adj=threshold \
  --tau=0.8 \
  --proj_dim=256 \
  --precompute_hidden_states=1 \
  --override_precompute=0 \
  --finetune_backbone=0 
```



## Evaluation on IMDB Benchmark

```bash
python main.py \
  --model_name_or_path='bert-base-uncased' \      # Any model from huggingface
  --decoder_cls_last_token=0 \    # Other choices: [1]
  --task=imdb \   
  --max_length=512 \
  --adaptive_length=0 \
  --epochs=3 \
  --batch_size=32 \
  --eval_batch_size=64 \
  --lr=2e-4 \
  --weight_decay=0.0 \
  --seed=42 \
  --verbose=1 \
  --pooling_method=glot \      # Other choices: [cls, mean, max, adapool]
  --gnn_type=gat \
  --scorer_hidden=128 \
  --gat_hidden_dim=256 \
  --num_layers=2 \
  --jk_mode=cat \
  --graph_adj=threshold \
  --tau=0.8 \
  --proj_dim=256 \
  --precompute_hidden_states=1 \
  --override_precompute=0 \
  --finetune_backbone=0 
```

## Evaluation on MTEB
First, download the [MS-MARCO Triplets](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz) dataset and extract it to `./data/msmarco-triplets.jsonl`. Then, train the GLOT sentence embedding model on this dataset using the following command:

```bash
python main.py \
  --model_name_or_path="mistralai/Mistral-7B-v0.1" \   
  --decoder_cls_last_token=1 \ 
  --task=embedding \
  --num_train_samples=full \   
  --max_length=128 \
  --adaptive_length=0 \
  --epochs=3 \
  --batch_size=32 \
  --eval_batch_size=64 \
  --lr=2e-5 \
  --weight_decay=0.0 \
  --seed=42 \
  --verbose=1 \
  --pooling_method=glot \      # Other choices: [adapool]
  --gnn_type=gat \
  --scorer_hidden=128 \
  --gat_hidden_dim=256 \
  --num_layers=4 \
  --jk_mode=cat \
  --graph_adj=threshold \
  --tau=0.3 \
  --proj_dim=512 \
  --precompute_hidden_states=1 \
  --override_precompute=0 \
  --finetune_backbone=0 
```

Next, modify the `checkpoint_path` value in `mteb_eval_config.yaml` (and `pooling_method` in case of `adapool`). Then, we use `wandb` to run the MTEB evaluation as follows:

```bash
wandb sweep mteb_eval_config.yaml --project GLOT
```

to obtain a `<sweep-id>` and launch the evaluation using
```bash
wandb agent <sweep-id>
```

Since there is no need for training the `mean`, `max` and `cls` baselines, (i) comment out the `checkpoint_path` variable in `mteb_eval_config.yaml`, (ii) change the value of `pooling_method`, and (iii) follow the same steps as above to obtain a `<sweep-id>` and launch the experiment.

## Diagnostic Stress Test

```bash
python diagnostic_stress_test.py \
    --batch_size=32 \
    --distractor_ratio=0.5 \  # Other choices: [0.2, 0.8, 0.9]
    --epochs=3 \
    --eval_batch_size=32 \
    --gat_hidden_dim=64 \
    --tau=0.6 \
    --lr=0.0001 \
    --model_name_or_path=mistralai/Mistral-7B-v0.1 \
    --num_layers=4  \
    --pooling_method=glot \
    --scorer_hidden=256 \
    --seed=0
```

## Cite Us
```bibtex
@InProceedings{mantri2026glot,
  author             = {Mantri, Krishna Sri Ipsit and Sch{\"o}nlieb, Carola-Bibiane and L{\"a}hner, Zorah and Eliasof, Moshe},
  title              = {Towards Improved Sentence Representations using Token Graphs},
  booktitle          = {The Fourteenth International Conference on Learning Representations (ICLR)},
  date               = {2026},
  pubstate           = {forthcoming},
  url                = {https://openreview.net/forum?id=stMX9KBhUI},
  abstract           = {Obtaining a single-vector representation from a Large Language Model's (LLM) token-level outputs is a critical step for nearly all sentence-level tasks. However, standard pooling methods like mean or max aggregation treat tokens as an independent set, discarding the rich relational structure captured by the model's self-attention layers and making them susceptible to signal dilution. To address this, we introduce GLOT, a lightweight, structure-aware pooling module that reframes pooling as relational learning followed by aggregation. Operating on the outputs of a frozen LLM, GLOT first constructs a latent token-similarity graph, then refines token representations with a graph neural network, and finally aggregates them using a readout layer. Experimentally, our approach is remarkably robust and efficient: on a diagnostic stress test where 90% of tokens are random distractors, GLOT maintains over 97% accuracy while baseline methods collapse. Furthermore, it competitive with state-of-the-art techniques on benchmarks like GLUE and MTEB with 20x fewer trainable parameters and speeds up the training time by over 100x compared with parameter-efficient fine-tuning methods. Supported by a theoretical analysis of its expressive power, our work shows that learning over token graphs is a powerful paradigm for the efficient adaptation of frozen LLMs.},   
  }
```