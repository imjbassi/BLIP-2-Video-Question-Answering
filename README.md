
# BLIP-2 Video Question Answering (VQA)

This repository implements a Video Question Answering system using Salesforce's [BLIP-2 FLAN-T5 XL](https://huggingface.co/Salesforce/blip2-flan-t5-xl) model. The system generates answers based on video content by extracting representative frames and applying multimodal prompts.

## 🧠 Project Overview

- **Model:** BLIP-2 FLAN-T5 XL  
- **Task:** Scene understanding and action reasoning from videos  
- **Approach:** Extract three key frames per video and prompt model for predictions  
- **Output:** JSON predictions compatible with EvalAI submissions

## 🚀 Quickstart

Clone and set up the repository:

```bash
git clone https://github.com/yourusername/blip2-vqa.git
cd blip2-vqa
pip install torch torchvision transformers opencv-python tqdm
````

Place validation videos and annotations into their respective folders (`valid_videos/videos/`, `mc_question_valid_annotations/mc_question_valid.json`), then run:

```bash
python vqa_inference.py
```

The predictions will be saved to `submission.json`.

## 📈 EvalAI Submission

Submit the generated `submission.json` file on EvalAI under the [Perception Test Challenge](https://eval.ai/web/challenges/challenge-page/2091):

* **Evaluation mode:** Select from `0-shot`, `8-shot`, `all-shot`, or `fine-tuned`
* **Visibility:** Public or private

## ⚙️ Tech Stack

* Python, PyTorch, Hugging Face Transformers, OpenCV, tqdm

* Efficient frame extraction and multimodal prompting
* Robust answer logging and checkpoint saving
* Direct EvalAI compatibility

