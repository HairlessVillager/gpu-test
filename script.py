from time import time
from random import randint

from tqdm import tqdm
from transformers import pipeline

nums = 10000
split = 8

with open("data/Shakespeare.txt") as f:
    text = f.read()
pipe = pipeline(
    "text-classification", model="TrustSafeAI/RADAR-Vicuna-7B", device="cuda:0"
)

total_start = time()

for i in tqdm(range(nums // split)):
    texts = []
    for j in range(split):
        start = randint(0, 9000000)
        end = start + randint(100, 800)
        sub_text = text[start:end]
        texts.append(sub_text)
    res = pipe(texts, batch_size=split)

total_end = time()
total_cost = total_end - total_start
print(f"{split=}, {total_cost:.6f} / {nums} = {total_cost / nums:.6f} s/item")
