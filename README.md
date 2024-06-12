# Reranking-of-TREC-Deep-Learning-2019-Track-Datasets-using-neural-method
This repository contains step-by-step guide to perform reranking documents of TREC DL 2019 Track Datasets using state-of-the-arts methods

* In the context of the [TREC Deep Learning (DL) Track task](https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.DL.pdf), **rerankers** play a pivotal role in refining search results by reordering them based on their relevance, thereby significantly improving the accuracy and usefulness of retrieved information.

## Introduction

Reranker is a lightweight, effective, and efficient package designed for training and deploying deep language model rerankers. These rerankers are essential for improving the performance of information retrieval (IR), question answering (QA), and various other natural language processing (NLP) pipelines. In the context of the TREC Deep Learning (DL) Track task, rerankers are used to refine search results by reordering them based on relevance, thus enhancing the accuracy and usefulness of the retrieved information.

## Results

The models used for tokenization are:

    bert-base
    deberta-v3
    naver/cocondenser-ensembledistil
    naver/cocondenser-selfdistil

With the help of these four models, the obtained results are summerized in the table as:

| Model                       | nDCG@10 | MRR    | MAP    |
|-----------------------------|---------|--------|--------|
| BERT-base                   | 0.5807  | 0.8682 | 0.2399 |
| DeBERTa-V3                  | 0.6197  | 0.9050 | 0.2497 |
| CoCondenser-EnsembleDistil  | 0.5924  | 0.8709 | 0.2398 |
| CoCondenser-SelfDistil      | 0.6207  | 0.8992 | 0.2498 |


## Task Details

The task involved performing reranking on the TREC DL-2019 track document datasets. The following sections outline the step-by-step process used to complete the task.

### Step 1: Setup

First, clone the Reranker repository and install the required libraries:

```sh
git clone https://github.com/luyug/Reranker.git
cd Reranker
pip install .
```

### Step 2: Download Datasets

Next, download the datasets required for the task from the following websites:

1. Download _hdct-marco-train.zip_ from the HDCT's [website](http://boston.lti.cs.cmu.edu/appendices/TheWebConf2020-Zhuyun-Dai/rankings/) and unzip this.
2. Download _msmarco-doctrain-qrels.tsv.gz , msmarco-docs.tsv.gz, msmarco-doctrain-queries.tsv.gz, msmarco-doctrain-top100.gz_ from [official website](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html) and unzip all.
3. Send me a request to download _test.d100.tsv_ .
4. Download official _qrel_ file from [trec website](https://trec.nist.gov/data/deep/2019qrels-pass.txt).

### Step 3: Preprocess Data

Create a _train_data_ directory where you will save the preprocessed large JSON data in tokenized form. Build Localized Training Data from top Ranking by the help of python script _create_train_data_from_ranking.py_ using command line code as:
```sh
mkdir -p train_data

for i in $(seq -f "%03g" 0 183); \
do \
    python3 create_train_data_from_ranking.py \
    --tokenizer_name <model-name> \
    --rank_file /path/to/bos/tmp11/zhuyund/hdct-marco-train-new/${i}.txt \
    --json_dir /path/to/train_data \
    --n_sample 10 \
    --sample_from_top 100 \
    --random \
    --truncate 512 \
    --qrel /path/to/msmarco-doctrain-qrels.tsv.gz \
    --query_collection /path/to/msmarco-doctrain-queries.tsv \
    --doc_collection /path/to/msmarco-docs.tsv; \
done
```
It generates a training set with localized negatives in json directory. 

### Step 4: Training

For the training all models on a single NVIDIA RTX A5000 GPU following command was used:

```sh
mkdir -p checkpoints

torchrun \
--nproc_per_node 1 searcher_and_scorer.py   \
--output_dir /path/to/checkpoints   \
--model_name_or_path <model-name>   \
--do_train   \
--save_steps 2000   \
--train_dir /path/to/train_data   \
--max_len 512   \
--fp16   \
--per_device_train_batch_size 2   \
--train_group_size 8   \
--gradient_accumulation_steps 2   \
--per_device_eval_batch_size 32   \
--warmup_ratio 0.1   \
--weight_decay 0.01   \
--learning_rate 1e-5   \
--num_train_epochs 2   \
--overwrite_output_dir   \
--dataloader_num_workers 8
```
Please note that You can run inference during training separtely by loading saved checkpoints. After training, the last few checkpoints are usually good.

### Step 5: Inference

First create a test output directory in which the following command will build ranking input:

```sh
mkdir -p testOutputF

python3 converter_topk_texts_into_json.py  \
    --file /path/to/test.d100.tsv \
    --save_to { path to testOutputF}/all.json \
    --generate_id_to { path to testOutputF}/ids.tsv \
    --tokenizer <model-name> \
    --truncate 512 \
    --q_truncate -1
```
Run inference with the help of generated input from last code using the trained model checkpoint using GPU with the following command:

```sh
torchrun \
  --nproc_per_node 1 searcher_and_scorer.py \
  --output_dir /path/to/testOutputF \
  --model_name_or_path {path to checkpoints}/checkpoint-X000 \
  --tokenizer_name <model-name> \
  --do_predict \
  --max_len 512 \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 8 \
  --pred_path { path to testOutputF}/all.json \
  --pred_id_file { path to testOutputF}/ids.tsv \
  --rank_score_path { path to testOutputF}/scores.txt
```

Given previous command will generate a scores.txt file in the same output directory. 

Further, convert the score to standard MS MARCO format to do trec_eval by the following command:

```sh
python3 converter_scores_into_marco.py \
--score_file {path to testOutputF}/scores.txt
```
