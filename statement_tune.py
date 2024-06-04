import argparse
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast,RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
#Autotokenize, automodel for seq class
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import evaluate
import wandb
import os




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Exp_name', type=str, default='roberta=base', help='Experiement Name of Run')
    parser.add_argument('--transformer', type=str, default='roberta-base', help='Transformer Model to be used')
    parser.add_argument('--cache', type=str, default='/scratch/afz225/.cache', help='Cache with Dataset')
    parser.add_argument('--save', type=str, default='./STTS_roberta-base', help='Save path for the final model')

    parser.add_argument('--tol', type=int, default=20, help='Tolerance')
    parser.add_argument('--test_size', type=int, default=0.1, help='Test data size')

    parser.add_argument('--tr_ep', type=int, default=2, help='Training Epochs')
    parser.add_argument('--tr_batch', type=int, default=8, help='Training Batch Size Per device')
    parser.add_argument('--ev_batch', type=int, default=16, help='Evaluation Batch Size Per device')
    parser.add_argument('--warmup', type=int, default=0.1, help='Warmup steps for learning rate scheduler')
    parser.add_argument('--lr', type=int, default=1e-06, help='Learning rate')
    parser.add_argument('--decay', type=int, default=0.01, help='Weight Decay')


    opts = parser.parse_args()
    return opts

'''
num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
    learning_rate=1e-06,
    weight_decay=0.01, 
'''



opts = parse_args()




EXPERIMENT_NAME = opts.model #"roberta-base"
CACHE_DIR = opts.cache #"/scratch/afz225/.cache"
MODEL_SAVE_PATH = opts.save #"./STTS_roberta-base"

os.environ["WANDB_PROJECT"]=f"{EXPERIMENT_NAME}_train"
wandb.login()
wandb.init(
    project=f"{EXPERIMENT_NAME}_train",
)

TRANSFORMER="roberta-base"
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER)

tolerance = 20
data = load_dataset('ashabrawy/STTS', cache_dir=CACHE_DIR)
train = data['train'].filter(lambda example: example["is_true"] is not None).filter(lambda example: len(tokenizer(example['statement'])['input_ids']) < 514+tolerance)

train = train.train_test_split(test_size=50000)['test']
train_statements, val_statements, train_labels, val_labels = train_test_split(train['statement'], train['is_true'], test_size=opts.test_size)

class StatementDataset(torch.utils.data.Dataset):
    def __init__(self, statements, labels):
        self.statements = statements
        self.labels = labels

    def __getitem__(self, idx):
        encodings = tokenizer(self.statements[idx], truncation=True, padding=True)
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = int(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

train_dataset = StatementDataset(train_statements, train_labels)
val_dataset = StatementDataset(val_statements, val_labels)

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries, which is in ids into text
    _, predictions = torch.max(torch.tensor(predictions), dim=1)


    return clf_metrics.compute(predictions=predictions, references=labels)
    

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)



training_args = TrainingArguments(
    output_dir=f'./{EXPERIMENT_NAME}-outputs',          # output directory
    num_train_epochs=opts.tr_ep,              # total number of training epochs
    per_device_train_batch_size=opts.tr_batch,  # batch size per device during training
    per_device_eval_batch_size=opts.ev_batch,   # batch size for evaluation
    warmup_ratio=opts.warmup,                # number of warmup steps for learning rate scheduler
    learning_rate=opts.lr,
    weight_decay=opts.decay,               # strength of weight decay
    logging_dir=f'./{EXPERIMENT_NAME}-logs',            # directory for storing logs
    logging_steps=1000,
    save_steps=1000,
    evaluation_strategy='steps',
    save_total_limit=2,
    load_best_model_at_end= True,
    metric_for_best_model='f1',
    report_to="wandb",
)

config = AutoConfig.from_pretrained(TRANSFORMER)
model = AutoModelForSequenceClassification.from_config(config)
#model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER,num_classes=2)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    compute_metrics=compute_metrics,
    eval_dataset=val_dataset,            # evaluation dataset
    data_collator=data_collator
)


trainer.train()

trainer.save_model(MODEL_SAVE_PATH)