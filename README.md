# BLIP-2 Video Question Answering (VQA)

This repository implements a Video Question Answering (VQA) pipeline using Salesforce’s [BLIP-2 FLAN-T5 XL](https://huggingface.co/Salesforce/blip2-flan-t5-xl) model. The system answers natural language questions about videos by extracting key frames, applying vision-language prompting, and generating JSON-formatted predictions compatible with EvalAI.

---

## Contents

* [Overview](#overview)
* [How It Works](#how-it-works)
* [Quickstart](#quickstart)
* [Directory Structure](#directory-structure)
* [Model and Dataset](#model-and-dataset)
* [Example Output](#example-output)
* [EvalAI Submission](#evalai-submission)
* [Dependencies](#dependencies)
* [Results](#results)
* [License](#license)

---

## Overview

| Component  | Description                                                     |
| ---------- | --------------------------------------------------------------- |
| Model      | BLIP-2 FLAN-T5 XL (Vision-Language Transformer)                 |
| Task       | Video Question Answering (VQA)                                  |
| Input      | `.mp4` video + associated multiple-choice question              |
| Output     | Predicted answer (`submission.json`)                            |
| Use Case   | Perception Test Challenge (EvalAI VQA benchmark)                |
| Evaluation | Multiple-choice accuracy score based on model-generated answers |

---

## How It Works

The system extracts representative frames from each input video, applies multimodal prompting with the BLIP-2 model, selects the most relevant answer from multiple-choice options, and compiles all results into a `submission.json` file for EvalAI evaluation.

### Visual Pipeline

![BLIP-2 VQA Pipeline](./A_flowchart_infographic_visually_illustrates_Video.png)

### Step-by-Step

1. **Frame Extraction**
   Three frames are extracted per video (start, middle, end) using OpenCV.

2. **Prompt Construction**
   Each question is paired with the extracted frames and formatted into a multimodal prompt.

3. **Answer Generation**
   The prompt is passed to BLIP-2, which generates a textual response using vision-language reasoning.

4. **Answer Matching**
   The model output is matched against multiple-choice options to select the closest valid answer.

5. **Result Logging**
   The answer is saved in a format compatible with EvalAI's submission interface.

---

## Quickstart

### Clone and Setup

```bash
git clone https://github.com/yourusername/blip2-vqa.git
cd blip2-vqa
pip install torch torchvision transformers opencv-python tqdm
```

### Add Data

* Place validation videos into: `valid_videos/videos/`
* Place question annotations into: `mc_question_valid_annotations/mc_question_valid.json`

### Run Inference

```bash
python vqa_inference.py
```

The output will be saved as `submission.json` in the current directory.

---

## Directory Structure

```bash
blip2-vqa/
│
├── vqa_inference.py               # Main pipeline script
├── submission.json                # Output predictions for EvalAI
├── valid_videos/
│   └── videos/                    # Raw .mp4 input videos
├── mc_question_valid_annotations/
│   └── mc_question_valid.json     # EvalAI question annotation format
├── A_flowchart_infographic_visually_illustrates_Video.png
└── README.md
```

---

## Model and Dataset

* **Model:** [BLIP-2 FLAN-T5 XL](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
* **Framework:** Hugging Face Transformers
* **Dataset:** EvalAI Perception Test Challenge validation set
* **Benchmark Format:** Each sample consists of a video and a multiple-choice question.

---

## Example Output

**Question:**

> What is the woman doing in the video?

**Predicted Answer:**

> She is dancing.

**JSON Output Format:**

```json
[
  {
    "question_id": "abcd1234",
    "answer": "She is dancing."
  },
  ...
]
```

---

## EvalAI Submission

Once `submission.json` is generated, upload it to the [Perception Test Challenge on EvalAI](https://eval.ai/web/challenges/challenge-page/2091):

* **Submission Modes:**

  * `0-shot`, `8-shot`, `all-shot`, or `fine-tuned`

* **Visibility:**

  * Public (for leaderboard) or Private

* **Evaluation Metric:**

  * Multiple-choice accuracy

---

## Dependencies

* Python 3.8+
* PyTorch
* Hugging Face Transformers
* OpenCV
* tqdm

Install all with:

```bash
pip install torch torchvision transformers opencv-python tqdm
```

---

## Results

| Evaluation Mode | Accuracy (example only) |
| --------------- | ----------------------- |
| 0-shot          | 47.3%                   |
| 8-shot          | 55.1%                   |
| All-shot        | 58.9%                   |

*Note: Actual results may vary depending on model checkpoint and prompt design.*

---

## License

This project is licensed for academic and non-commercial use. See `LICENSE` for details.
