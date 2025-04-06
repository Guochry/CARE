# CARE
This repository is the official implementation of paper: **[CARE: Aligning Language Models for Regional Cultural Awareness](https://arxiv.org/pdf/2311.04072.pdf)**. 


## Quick Start
You should clone the TextBox repository and follow its instructions.
```
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
bash install.sh
```

## SPA Datasets

You can download SPA datasets in: https://huggingface.co/datasets/RUCAIBox/Erya-dataset. You should download datasets such as xint in it and place them in the `dataset` folder.



## Alignment tuning and Inference
After setting up the environment, you can either use FIGA model in the zero-shot scenario, or train it on your own dataset from scratch.

### Inference
We have released FIGA model in: [https://huggingface.co/RUCAIBox/Erya](https://huggingface.co/RUCAIBox/Erya), which you can use directly as below.

```
from transformers import BertTokenizer, CPTForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained("RUCAIBox/Erya")
model = CPTForConditionalGeneration.from_pretrained("RUCAIBox/Erya")

input_ids = tokenizer("安世字子孺，少以父任为郎。", return_tensors='pt')
input_ids.pop("token_type_ids")

pred_ids = model.generate(max_new_tokens=256, **input_ids)
print(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
```

### Tuning
To align your own model on another dataset from scratch, you can go as below.

```
python run_textbox.py --model=CPT --dataset=[dataset] --model_path=RUCAIBox/Erya4FT --epochs=[epoch_nums]
```


## Acknowledgment
Please cite the following paper if you find our code or data helpful.

```
@article{guo2023beyond,
  title={CARE: Aligning Language Models for Regional Cultural Awareness},
  author={Guo, Geyang and Naous, Tarek and Wakaki, Hiromi and Nishimura, Yukiko and Mitsufuji, Yuki and Ritter, Alan and Xu, Wei},
  journal={arXiv preprint arXiv:2311.04072},
  year={2025}
}
```


