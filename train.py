import gc
import os
import json
import time
import argparse
import shutil
import subprocess
import warnings
import collections
from glob import glob

from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub.hf_api import HfFolder, HfApi
from huggingface_hub.repository import Repository

from dataset import CustomDataset, truncate_text
from models import MLUBModel
from train_utils import compute_metrics, get_kfold
from optimizers import create_optimizer_roberta_large

warnings.filterwarnings("ignore")

GIT_USER = "bayartsogt"
GIT_EMAIL = "bayartsogtyadamsuren@icloud.com"

if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description="Simple example of training script.")
    # parser.add_argument("--fold", type=int, default=0, help="If passed, will train only this fold.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds to train")

    # model selection
    parser.add_argument("--model-name-or-path", type=str, default="roberta-base", help="Hugginface base model name")

    # hyperparameter
    parser.add_argument("--num-epochs", type=int, default=15, help="Number of Epoch")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning Rate")
    parser.add_argument("--warmup-steps", type=int, default=400, help="Number of warm up steps")
    parser.add_argument("--roberta-large-optimizer", action="store_true", default=False, help="Use specific optimizer for Large models")
    parser.add_argument("--truncate-length", type=int, default=30, help="Effective length of the train input")
    parser.add_argument("--max-len", type=int, default=150, help="Effective length of the train input")

    # dataloader
    parser.add_argument("--seed", type=int, default=42, help="If passed, seed will be used for reproducability")

    # huggingface hub
    parser.add_argument("--push-to-hub", action="store_true", default=False, help="model will be saved in huggingface hub")
    parser.add_argument("--submit-to-kaggle", action="store_true", default=False, help="submission.csv will be submitted to kaggle")
    args = parser.parse_args()

    output_dir = "mlub-" + \
            f"{args.model_name_or_path.split('/')[0]}-" + \
            f"tr{args.truncate_length}"

    # ----------------------------- HF API --------------------------------
    if args.push_to_hub:
        hf_token = HfFolder.get_token(); api = HfApi()
        repo_link = api.create_repo(token=hf_token, name=output_dir, exist_ok=True, private=True)
        repo = Repository(local_dir=output_dir, clone_from=repo_link, use_auth_token=hf_token, git_user=GIT_USER, git_email=GIT_EMAIL)
        print("[success] configured HF Hub to", output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/training_arguments.json", "w") as writer:
        json.dump(vars(args), writer, indent=4)
        print(f"[success] wrote training args to {output_dir}")

    # ----------------------------- TRAINING ARGUMENTS --------------------------------
    training_args = TrainingArguments(
        output_dir="/content/output", # this will be replaced
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*4,
        dataloader_num_workers=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=False,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        save_total_limit=args.num_epochs,
        # report_to = 'wandb',
        # run_name = 'ulaanbal train exp0'
    )

    # ----------------------------- DATA --------------------------------
    print("loading data")
    BASE_DIR = "data/preprocessed"
    df_submission = pd.read_csv(f"{BASE_DIR}/submission.csv")
    df_train = pd.read_csv(f"{BASE_DIR}/train.csv")
    df_test = pd.read_csv(f"{BASE_DIR}/test.csv")
    df_synset_meaning = pd.read_csv(f"{BASE_DIR}/synset_meaning.csv")
    df_synset_meaning["word_len"] = df_synset_meaning.word.apply(len)
    print(df_train.shape, df_test.shape, df_submission.shape, df_synset_meaning.shape)

    # ----------------------------- PREPROCESS --------------------------------
    dict_synset_meaning = collections.defaultdict(dict)
    for row in df_synset_meaning.itertuples():
        dict_synset_meaning[row.word][row.synset_id] = row.meaning.lower()
    synset_id2word = {row.synset_id:row.word for row in df_synset_meaning.itertuples()}
    unique_synset = set(df_synset_meaning.word.unique().tolist())

    # truncate text
    df_train = truncate_text(df_train, effective_len=args.truncate_length)
    df_test = truncate_text(df_test, effective_len=args.truncate_length)

    ids = sorted(df_synset_meaning.synset_id.unique().tolist())
    index2id, id2index = {i:id for i, id in enumerate(ids)}, {id:i for i, id in enumerate(ids)}
    num_labels = len(index2id)

    df_train["synset_index"] = df_train.synset_id.map(id2index)
    df_train = get_kfold(df_train, num_folds=args.num_folds, random_state=args.seed)

    # ----------------------------- TOKENIZER --------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(output_dir)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"TOKENIZERS_PARALLELISM = {os.environ['TOKENIZERS_PARALLELISM']}")

    df_test["synset_index"] = 0
    test_dataset = CustomDataset(df_test, tokenizer, max_len=args.max_len)

    predictions = []
    oof = np.zeros((df_train.shape[0], num_labels))

    for fold in range(args.num_folds):
        train = df_train.query("fold!=@fold")
        valid = df_train.query("fold==@fold")

        train_dataset = CustomDataset(train, tokenizer, max_len=args.max_len)
        valid_dataset = CustomDataset(valid, tokenizer, max_len=args.max_len)

        training_args.output_dir = f"{output_dir}/fold_{fold}"

        model = MLUBModel(
            args.model_name_or_path,
            num_labels=num_labels
        )
        optimizer = None
        if args.roberta_large_optimizer:
            optimizer = create_optimizer_roberta_large(model=model, learning_rate=args.learning_rate)

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            optimizers=(optimizer, None)
        )

        train_output = trainer.train()
        print("fold", fold, "=>", trainer.evaluate(valid_dataset))
        
        y_hat = trainer.predict(test_dataset)
        predictions.append(y_hat.predictions)

        y_hat = trainer.predict(valid_dataset)
        oof[valid.index] = y_hat.predictions
        
        torch.save(trainer.model.state_dict(), f"{output_dir}/model_{fold}.bin")

        shutil.rmtree(f"{output_dir}/fold_{fold}"); print(f"Removed {output_dir}/fold_{fold}")

    df_train["pred_synset_index"] = np.array(oof.argmax(1), dtype=np.int32)
    df_train["pred_synset_id"] = df_train.pred_synset_index.map(index2id)
    df_train.drop("text_clean", axis=1).to_csv(f"{output_dir}/oof.csv", index=False)
    print("************")
    oof_accuracy = accuracy_score(df_train.synset_index.values, oof.argmax(1))
    print("OOF Acc:", oof_accuracy)

    np.save(f"{output_dir}/oof.npy", oof)
    np.save(f"{output_dir}/predictions.npy", np.array(predictions).mean(0))
    df_submission["synset_index"] = np.array(predictions).mean(0).argmax(-1)
    df_submission.loc[:, "synset_id"] = df_submission.loc[:,"synset_index"].map(index2id)
    df_submission.head(1)

    df_submission.drop("synset_index", axis=1).to_csv(f"{output_dir}/submission.csv", index=False)

    if args.push_to_hub:
        repo.git_pull() # get updates first
        commit_link = repo.push_to_hub(commit_message=f"AVG: => ACC: {oof_accuracy:.3f}") # then push
        print("[success] UPLOADED TO HUGGINGFACE HUB", commit_link)
        

    if args.submit_to_kaggle:
        print("submitting to kaggle")
        process = subprocess.run([
            "kaggle",
            "competitions",
            "submit",
            "-c",
            "muis-challenge",
            "-f",
            f"{output_dir}/submission.csv",
            "-m",
            f"Acc{oof_accuracy:.2f}"
        ], capture_output=True)

        print(process)
    
    print("[success] TIME SPENT: %.3f min" % ((time.time()-start_time) / 60))