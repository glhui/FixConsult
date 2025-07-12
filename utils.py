import argparse

import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

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

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def process_patch(patch_text):
    """
    对单个 patch 文本进行处理：
    - 每行以 '+' 开头的替换为 <ADDED>，以 '-' 开头的替换为 <REMOVED>
    - 其他行原样保留
    """
    processed_lines = []
    for line in patch_text.splitlines():
        if line.startswith('+'):
            processed_line = "<ADDED> " + line[1:].strip()
        elif line.startswith('-'):
            processed_line = "<REMOVED> " + line[1:].strip()
        else:
            processed_line = line.strip()
        processed_lines.append(processed_line)
    return "\n".join(processed_lines)

def build_input_text(func, patch, defect_type, strategy):
    """
    构造输入文本，将函数代码和补丁拼接为一个字符串，
    格式：<CODE> 函数代码 </CODE> <PATCH> 处理后的补丁 </PATCH>
    """
    func = func.strip() if isinstance(func, str) else ""
    patch = patch.strip() if isinstance(patch, str) else ""
    processed_patch = process_patch(patch) if patch != "" else ""
    if strategy == "final":
        return f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func} </CODE> <PATCH> {processed_patch} </PATCH>"
    elif strategy == "no_patch":
        return f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func} </CODE>"
    elif strategy == "no_mask":
        return f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func} </CODE> <PATCH> {processed_patch} </PATCH>"
    elif strategy == "no_label":
        return f"<CODE> {func} </CODE> <PATCH> {processed_patch} </PATCH>"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def create_ra_masks(texts, tokenizer):
    docs = [nlp(text) for text in texts]
    word_lists = [[token.text for token in doc] for doc in docs]
    # encoding = tokenizer(text_target=word_lists, is_split_into_words=True, padding=True, truncation=True, max_length=100)
    encoding = tokenizer(text=word_lists, is_split_into_words=True, padding=True, truncation=True, max_length=100)
    ra_masks = []

    for i, doc in enumerate(docs):
        # Initialize all ra masks to 0 (ignore all tokens initially)
        ra_mask = [0]*len(encoding['attention_mask'][i])

        word_ids = encoding.word_ids(batch_index=i)  # Get word_ids for the specific batch index

        # Identify verbs and their objects using spaCy's parse
        for token in doc:
            if token.pos_ == 'VERB':
                verb_idx = token.i
                for child in token.children:
                    if child.dep_ in ['dobj', 'iobj', 'pobj', 'nsubjpass']:
                        obj_idx = child.i
                        # Find tokens corresponding to verb and object
                        verb_tokens = [idx for idx, word_id in enumerate(word_ids) if word_id == verb_idx]
                        obj_tokens = [idx for idx, word_id in enumerate(word_ids) if word_id == obj_idx]

                        # Set ra mask to 1 for verb-object tokens
                        for vt in verb_tokens:
                            ra_mask[vt] = 1  # Attend to these tokens
                        for ot in obj_tokens:
                            ra_mask[ot] = 1  # Attend to these tokens

        ra_masks.append(ra_mask)

    return encoding["input_ids"], ra_masks


def defect_type_detection(texts: str):
    pass