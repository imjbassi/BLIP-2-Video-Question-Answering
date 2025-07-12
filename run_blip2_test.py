import os
import json
import zipfile
import logging
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(filename="process.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on device: {device}")

model_name = "Salesforce/blip2-flan-t5-xl"
logging.info(f"Loading BLIP-2 model '{model_name}' for test inference...")
processor = AutoProcessor.from_pretrained(model_name)
if device.type == "cuda":
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
else:
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)
model.to(device)
model.eval()
logging.info("Model loaded.")

frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

videos_zip_path = "test_videos.zip"
ann_zip_path = "test_annotations.zip"
try:
    videos_zip = zipfile.ZipFile(videos_zip_path, 'r')
except FileNotFoundError:
    logging.error(f"Video zip file {videos_zip_path} not found.")
    raise
try:
    ann_zip = zipfile.ZipFile(ann_zip_path, 'r')
except FileNotFoundError:
    logging.error(f"Annotation zip file {ann_zip_path} not found.")
    raise

# Load test questions annotations (no answers provided in test)
test_ann = None
for name in ann_zip.namelist():
    if name.endswith(".json") and ("mc_question" in name or "multiple_choice" in name or "test" in name):
        logging.info(f"Loading test annotations from {name}.")
        with ann_zip.open(name) as f:
            test_ann = json.load(f)
        break
if test_ann is None:
    logging.error("Test question annotations not found in the annotation zip.")
    raise RuntimeError("Failed to locate test MC-QA annotations in zip.")

# Group annotations by video
if isinstance(test_ann, dict):
    video_items = list(test_ann.items())
elif isinstance(test_ann, list):
    grouped = {}
    for q in test_ann:
        vid = q.get("video_id") or q.get("video") or q.get("video_uid")
        if vid is None:
            continue
        grouped.setdefault(vid, []).append(q)
    video_items = list(grouped.items())
else:
    logging.error("Unrecognized test annotation format.")
    raise RuntimeError("Annotation format not recognized.")

submission = {}

def process_video(item):
    video_id, questions = item
    logging.info(f"Processing video {video_id} ({len(questions)} questions).")
    frame_paths = {
        "first": os.path.join(frames_dir, f"{video_id}_frame_first.jpg"),
        "middle": os.path.join(frames_dir, f"{video_id}_frame_middle.jpg"),
        "last": os.path.join(frames_dir, f"{video_id}_frame_last.jpg")
    }
    frames_extracted = {}
    if all(os.path.exists(p) for p in frame_paths.values()):
        logging.info(f"Using cached frames for video {video_id}.")
        for pos, path in frame_paths.items():
            try:
                frames_extracted[pos] = Image.open(path).convert("RGB")
            except Exception as e:
                logging.error(f"Failed to open cached frame {path}: {e}")
                frames_extracted[pos] = None
    else:
        try:
            video_file = videos_zip.open(f"{video_id}.mp4")
        except KeyError:
            logging.error(f"Video file {video_id}.mp4 not found in zip.")
            return (video_id, [])
        tmp_video_path = os.path.join(frames_dir, f"{video_id}.mp4")
        with open(tmp_video_path, "wb") as vf:
            vf.write(video_file.read())
        video_file.close()
        import cv2
        cap = cv2.VideoCapture(tmp_video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video {video_id}.")
            cap.release()
            os.remove(tmp_video_path)
            return (video_id, [])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frames = []
            success, frame = cap.read()
            while success:
                frames.append(frame); success, frame = cap.read()
            frame_count = len(frames)
        first_idx = 0
        last_idx = frame_count - 1 if frame_count > 0 else 0
        middle_idx = frame_count // 2 if frame_count > 1 else first_idx
        target_indices = {"first": first_idx, "middle": middle_idx, "last": last_idx}
        for pos, idx in target_indices.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Failed to read {pos} frame for video {video_id}.")
                frames_extracted[pos] = None
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                frames_extracted[pos] = img
                try:
                    img.save(frame_paths[pos], "JPEG")
                except Exception as e:
                    logging.error(f"Could not save {pos} frame for {video_id}: {e}")
        cap.release()
        os.remove(tmp_video_path)
    if not frames_extracted:
        logging.error(f"No frames for video {video_id}.")
        return (video_id, [])
    predictions = []
    for q in questions:
        q_id = q.get("id")
        question_text = q.get("question")
        options = q.get("options", [])
        if not question_text or len(options) < 3:
            logging.warning(f"Skipping question {q_id} due to missing data.")
            continue
        optA, optB, optC = options[0], options[1], options[2]
        prompt = f"Question: {question_text} Options: (A) {optA} (B) {optB} (C) {optC} Answer:"
        frame_answers = []
        for pos, img in frames_extracted.items():
            if img is None: 
                continue
            try:
                inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
                gen_ids = model.generate(**inputs, max_new_tokens=20)
                out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                logging.error(f"Inference error on video {video_id} q {q_id} ({pos}): {e}")
                out_text = ""
            frame_answers.append(out_text)
            logging.info(f"Video {video_id} Q{q_id} {pos} frame answer: \"{out_text}\"")
        chosen_idx = 0
        if frame_answers:
            norm_ans = [ans.lower().strip() for ans in frame_answers]
            # Check for explicit option letter in answer
            for ans in norm_ans:
                if ans in ["a", "(a)", "option a", "answer a"]: chosen_idx = 0; break
                if ans in ["b", "(b)", "option b", "answer b"]: chosen_idx = 1; break
                if ans in ["c", "(c)", "option c", "answer c"]: chosen_idx = 2; break
            if len(frame_answers) > 1:
                counts = [0,0,0]
                for ans in norm_ans:
                    for i, opt in enumerate(options):
                        if opt.lower() == ans:
                            counts[i] += 1
                if max(counts) > len(norm_ans)//2:
                    chosen_idx = counts.index(max(counts))
                else:
                    mid_ans = norm_ans[len(norm_ans)//2]  # default to middle frame's answer on tie
                    for i, opt in enumerate(options):
                        if opt.lower() == mid_ans:
                            chosen_idx = i
                            break
        if chosen_idx < 0 or chosen_idx >= len(options):
            chosen_idx = 0
        chosen_text = options[chosen_idx]
        logging.info(f"Final prediction for video {video_id} question {q_id}: Option {chosen_idx} \"{chosen_text}\"")
        predictions.append({
            "id": q_id,
            "answer_id": chosen_idx,
            "answer": chosen_text
        })
    return (video_id, predictions)

max_workers = min(4, len(video_items))
logging.info(f"Processing test videos with {max_workers} parallel workers...")
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_video, item) for item in video_items]
    for future in futures:
        vid, preds = future.result()
        if preds is not None:
            submission[vid] = preds

with open("submission_test.json", "w") as outf:
    json.dump(submission, outf)
logging.info("Saved test predictions to submission_test.json")
print("Submission file 'submission_test.json' generated.")
