import argparse
import os

import pandas as pd
from bert_score import BERTScorer
from evaluate import load
from tqdm import tqdm

from bar import behavioral_acc
import sys
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="config"
    )
    parser.add_argument(
        "--llm_type", type=str, required=True,
    )
    parser.add_argument(
        "--strategy", type=str, required=True,
    )
    return parser.parse_args()

def main():
    args = parse_args()

    bleu = load("sacrebleu")
    rouge = load("rouge")

    df = pd.read_csv("/data/hugang/GlhCode/VulAdvisor/data/data/test.csv")
    refs = list(df["suggestion"])

    predictions_dir = f"./predictions/{args.llm_type}/{args.strategy}"
    print(f"[info] predictions_dir: {predictions_dir}=-================================================================")
    for file in tqdm(os.listdir(predictions_dir), desc="Evaluating"):
        if file.endswith(".out"):
            with open(os.path.join(predictions_dir, file)) as f:
                lines = [line for line in f.readlines() if line.strip()]


            print(f"The score of {file}===================================================")
            scores = [behavioral_acc(r, p) for r, p in zip(refs, lines)]
            print(sum(scores)/len(scores))

            bleu_score = bleu.compute(predictions=lines, references=[[ref] for ref in refs])["score"]
            rouge_score = rouge.compute(predictions=lines, references=refs)

            print(bleu_score, rouge_score)
            # print(f"BLEU: {bleu_score:.4f}, ROUGE-1: {rouge_score['rouge1'].mid.fmeasure:.4f}, ROUGE-L: {rouge_score['rougeL'].mid.fmeasure:.4f}")
            scorer = BERTScorer(model_type='bert-base-uncased')
            P, R, F1 = scorer.score(lines, refs)
            print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
            print("==================================================================================================")

if __name__ == '__main__':
    main()