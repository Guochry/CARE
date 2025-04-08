# CARE <img src="fig/Hugging.jpg" alt="CARE Banner" width="50"/>
This repository is the official implementation of our paper: **[CARE: Aligning Language Models for Regional Cultural Awareness](https://arxiv.org/pdf/2504.05154)**. 



## CARE Resource

You can download CARE resource in: [https://huggingface.co/datasets/geyang627/CARE](https://huggingface.co/datasets/geyang627/CARE), which includes multilingual responses with human preferences on culture-specific questions.

Specifically, the `question` field is the culture-specific question, the `response` field contains responses generated by LLM (e.g. Qwen2.5-7B-Instruct) or written by human, the `culture_type` field contains cultural context category, `associated_culture` field contains the associated culture, and `rating` field contains the rating given by human on a 10-point scale, For a detailed description of the construction process of CARE, please refer to our paper.


## Culturally Aligned Models
We have released the culturally aligned models using CARE in: [geyang627/CARE](https://huggingface.co/collections/geyang627/care-67f42f022663b58f9ba10aea), which includes multilingual responses with human preferences on culture-specific questions.
You can use them directly as below.

```
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

model = LLM(model="geyang627/care-chinese-qwen2.5-7b", tensor_parallel_size=torch.cuda.device_count(), dtype="auto", trust_remote_code=True, max_model_len=2048)

tokenizer = AutoTokenizer.from_pretrained("geyang627/care-chinese-qwen2.5-7b", use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=256)
outputs = model.generate(["为什么中国人不喜欢数字4？"], sampling_params)
print(outputs[0].outputs[0].text)
```

## Evaluation
To evaluate model's cultural awareness with CARE, you can assess our test set in [geyang627/CARE-eval](https://huggingface.co/datasets/geyang627/CARE-eval) and use our prompt in the directory `prompts`.



## Acknowledgment
Please cite the following paper if you find our code or data helpful.

```
@article{guo2025care,
  title={CARE: Aligning Language Models for Regional Cultural Awareness},
  author={Guo, Geyang and Naous, Tarek and Wakaki, Hiromi and Nishimura, Yukiko and Mitsufuji, Yuki and Ritter, Alan and Xu, Wei},
  journal={arXiv preprint arXiv:2504.05154},
  year={2025}
}
```


