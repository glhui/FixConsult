import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import threading
import httpx

clients = [
    OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="Your-key", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="Your-key", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="Your-key", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ]


def build_prompt(repair_note: str, diff: str) -> str:
    prompt = f"""
You are a senior C/C++ security and quality expert.
Based on the following "Repair Note" and "Patch (before→after)" content, return only the most likely defect type(s) without any reasoning. 
Be as simple, accurate, and clear as possible. If there are multiple defect types, separate them with commas; if none, return ‘others’.

Available defect types:
- mem_leak        (Memory Leak)
- null_deref      (Null Pointer Dereference)
- out_of_bound    (Array Out-of-Bounds Access)
- use_after_free  (Use-After-Free)
- data_race       (Concurrency Data Race)
- logic_error     (Logic Error / Incomplete Implementation)
- vuln_api_use    (Use of Vulnerable/Unsafe API)
- resource_leak   (Resource Leak - File/Handle)
- assertion_fail  (Assertion Trigger)
- update          (Update for read or other reasons)

---
Repair Note:
{repair_note}

Patch (Diff before→after):
```diff
{diff}
```

Defect Type:
"""
    return prompt

key_name = ["kong", "cs", "wojie", "pangpang", "hu", "huang"]

# Function to call the OpenAI API and extract defect type
def get_defect_type(repair_note: str, diff: str, client, model = "deepseek-v3") -> str:
    prompt = build_prompt(repair_note, diff)
    # 创建聊天完成
    completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model=f"{model}",
    messages=[
        {"role": "system", "content": "You classify defect types."},
        {"role": "user", "content": f"{prompt}"},
    ],
)
    # Extract and clean label
    # print(completion.model_dump_json())
    label = completion.choices[0].message.content.strip()
    # print(label)
    return label

def process_partition(df, start_idx, end_idx, index, model="deepseek-v3"):
    for idx in tqdm(range(start_idx, end_idx), desc=f"Thread-{start_idx}-{end_idx}"):
        try:
            suggestion = df.at[idx, 'suggestion']
            patch = df.at[idx, 'patch']
            df.at[idx, 'defect_type'] = get_defect_type(suggestion, patch, clients[index], model)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            print(f"[Warn Info]: start_idx: {start_idx}, end_idx: {end_idx}, error happened at index {idx} and name is {key_name[index]}")

def multithread_process_df(df, num_threads=5, model_name = "deepseek-v3"):
    threads = []
    start_index = 0
    total_rows = len(df)
    remain_rows = len(df) - start_index
    
    import math
    chunk_size = math.ceil(remain_rows / num_threads)

    for i in range(num_threads):
        start_idx = start_index + i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        thread = threading.Thread(target=process_partition, args=(df, start_idx, end_idx, i % len(clients), model_name))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return df

# Main processing
def main():
    # Read input CSV; expect columns 'repair_note' and 'diff'
    df_test = pd.read_csv("data/test.csv")
    df_training = pd.read_csv("data/train.csv")
    df_validation = pd.read_csv("data/valid.csv")
    df = [df_training]
    model: list[str] = ["qwen-max"]
    for m in model:
        print(f"Processing with model: {m}")
        for d, t in zip(df, ["train"]):
            multithread_process_df(d, num_threads=len(clients), model_name=m)
            os.makedirs(f"data/{m}", exist_ok=True)
            d.to_csv(f"data/{m}/label_{t}.csv", index=False)
        print("========================================================================================")
    

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Label defects in patch suggestions.")
    # parser.add_argument("--input", required=True, help="Path to input CSV with columns 'repair_note' and 'diff'.")
    # parser.add_argument("--output", required=True, help="Path to output CSV to save with new 'defect_type' column.")
    # args = parser.parse_args()
    main()
