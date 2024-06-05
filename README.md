# UGRIP_MST
Multilingual Statement Tuning

## Running Evaluation Script
`python evalscript.py --model ashabrawy/ST-roberta-base --tokenizer roberta-base`

## Datasets, Trained Models Repo in HuggingFace
https://huggingface.co/mbzuai-ugrip-statement-tuning

## Running the Training Script
`python3 statement_tune.py --Exp_name "XLMR" --transformer "xlm-roberta-base" --save "./XLMR2"`
The Base statement to train the XLMR model on english dataset, can add hyperparameter flags as needed. 
Supported flags:
--tol
--test_size
--tr_ep
--tr_batch
--ev_batch
--warmup
--lr
--decay
