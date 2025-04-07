# CARE
This repository is the official implementation of paper: **[CARE: Aligning Language Models for Regional Cultural Awareness](https://arxiv.org/pdf/2311.04072.pdf)**. 


## CARE Datasets

You can download CARE dataset in: https://huggingface.co/datasets/RUCAIBox/Erya-dataset.



## CARE model Inference
After setting up the environment, you can either use FIGA model in the zero-shot scenario, or train it on your own dataset from scratch.

### Inference
We have released CARE model in: [https://huggingface.co/RUCAIBox/Erya](https://huggingface.co/RUCAIBox/Erya), which you can use directly as below.

```
from transformers import BertTokenizer, CPTForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained("RUCAIBox/Erya")
model = CPTForConditionalGeneration.from_pretrained("RUCAIBox/Erya")

input_ids = tokenizer("安世字子孺，少以父任为郎。", return_tensors='pt')
input_ids.pop("token_type_ids")

pred_ids = model.generate(max_new_tokens=256, **input_ids)
print(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
```

## Evaluation
To evaluate the model with CARE setup, you can assess our test set and use our prompt.



## Acknowledgment
Please cite the following paper if you find our code or data helpful.

```
@article{guo2025care,
  title={CARE: Aligning Language Models for Regional Cultural Awareness},
  author={Guo, Geyang and Naous, Tarek and Wakaki, Hiromi and Nishimura, Yukiko and Mitsufuji, Yuki and Ritter, Alan and Xu, Wei},
  journal={arXiv preprint arXiv:2311.04072},
  year={2025}
}
```


