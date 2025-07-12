import os
import json
import torch
import cv2
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

# Setup logging
logging.basicConfig(filename='inference.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)
model.to(device)

# Local paths
VIDEO_DIR = os.path.join("valid_videos", "videos")
ANNOTATION_FILE = os.path.join("mc_question_valid_annotations", "mc_question_valid.json")
OUTPUT_FILE = "submission.json"

# Load annotations
with open(ANNOTATION_FILE, 'r') as f:
    annotations = json.load(f)

# Resume from existing predictions
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        submission = json.load(f)
else:
    submission = {}

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = [0, frame_count // 2, frame_count - 1]
    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def predict_answer(question, options, frames):
    answers = []
    for frame in frames:
        prompt = (
            f"You are given a frame from a video. "
            f"Question: {question} Choose the correct option: "
            f"(A) {options[0]}, (B) {options[1]}, (C) {options[2]}. Answer:"
        )
        inputs = processor(images=frame, text=prompt, return_tensors="pt").to(
            device, torch.float16 if device.type == "cuda" else torch.float32
        )
        out = model.generate(**inputs, max_new_tokens=10)
        answer = processor.decode(out[0], skip_special_tokens=True).strip().lower()
        print(f"\nPrompt: {prompt}\nGenerated Answer: {answer}")
        logging.info(f"Prompt: {prompt} | Answer: {answer}")
        answers.append(answer)

    letter_map = {"a": 0, "b": 1, "c": 2}
    counts = [0, 0, 0]
    for answer in answers:
        if "(" in answer and ")" in answer:
            answer = answer.strip("()")
        answer = answer.strip().lower()
        if answer in letter_map:
            counts[letter_map[answer]] += 1
        else:
            for i, option in enumerate(options):
                if option.lower() in answer or answer in option.lower():
                    counts[i] += 1
    best = max(range(3), key=lambda i: counts[i])
    return ["A", "B", "C"][best]

# Run inference
for video_id, video_data in tqdm(annotations.items()):
    if video_id in submission:
        continue

    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        logging.warning(f"Missing video file: {video_id}")
        continue
    if not isinstance(video_data, dict) or "mc_question" not in video_data:
        logging.warning(f"Skipping {video_id}: no mc_question data")
        continue

    frames = extract_frames(video_path)
    questions = video_data["mc_question"]
    submission[video_id] = []

    for i, q in enumerate(questions):
        if not isinstance(q, dict) or "question" not in q or "options" not in q:
            logging.warning(f"Skipping malformed question at {video_id}[{i}]: {q}")
            continue
        pred = predict_answer(q["question"], q["options"], frames)
        submission[video_id].append(pred)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"[Saved] {video_id} → {len(submission[video_id])} answers")

print("✅ All done. Final predictions saved to", OUTPUT_FILE)

# Optional manual test
print("\n🧪 Manual test on single frame...")
test_vid = os.path.join(VIDEO_DIR, "video_8913.mp4")
if os.path.exists(test_vid):
    frame = extract_frames(test_vid)[1]
    test_q = "Where is the person?"
    test_opts = ["kitchen", "outdoors", "bedroom"]
    print("Prediction:", predict_answer(test_q, test_opts, [frame]))
else:
    print("Test video not found.")
