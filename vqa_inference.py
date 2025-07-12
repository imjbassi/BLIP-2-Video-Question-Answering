import os
import json
import argparse
import logging
from zipfile import ZipFile

import torch
from PIL import Image
from tqdm import tqdm

from transformers import Blip2Processor, Blip2ForConditionalGeneration


def extract_frames(video_path, num_frames):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        logging.error(f"Video has no frames: {video_path}")
        cap.release()
        return []
    # Determine frame indices to sample
    indices = [int(i * frame_count / float(num_frames)) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed reading frame {idx} from {video_path}")
            continue
        # Convert BGR to RGB and to PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def load_annotations(ann_path):
    # Load multiple-choice question annotations from a JSON file or zip file. Returns a list of question dicts.
    data = None
    if ann_path.lower().endswith('.zip'):
        try:
            with ZipFile(ann_path, 'r') as z:
                # Assume there is a single JSON file inside
                json_files = [f for f in z.namelist() if f.endswith('.json')]
                if not json_files:
                    logging.error("No JSON file found in annotation zip.")
                    return []
                with z.open(json_files[0]) as f:
                    data = json.load(f)
        except Exception as e:
            logging.error(f"Failed to open annotation zip {ann_path}: {e}")
            return []
    else:
        with open(ann_path, 'r') as f:
            data = json.load(f)
    if isinstance(data, dict):
        if 'questions' in data:
            return data['questions']
        if 'annotations' in data:
            return data['annotations']
        if isinstance(data, list):
            return data
    return data if data else []

def main():
    parser = argparse.ArgumentParser(description="BLIP2 VQA Validation Script (Multiple-Choice Video QA)")
    parser.add_argument("--video_dir", required=True, help="Directory containing validation videos")
    parser.add_argument("--annotations", required=True, help="Path to mc_question_valid_annotations.json or .zip")
    parser.add_argument("--output", default="submission.json", help="Output JSON file for predictions")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to sample per video")
    parser.add_argument("--device", default="cuda", help="Device for model (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log_level)

    # Load annotations
    questions = load_annotations(args.annotations)
    if not questions:
        logging.error("No questions loaded; exiting.")
        return
    logging.info(f"Loaded {len(questions)} questions from annotations.")

    # Group questions by video_id for efficiency
    q_by_video = {}
    for q in questions:
        vid = q.get("video_id") or q.get("video")
        if vid is None:
            logging.warning(f"Question missing video_id: {q.get('question_id','UNKNOWN')}")
            continue
        q_by_video.setdefault(vid, []).append(q)

    # Load BLIP2 model and processor
    model_name = "Salesforce/blip2-flan-t5-xl"
    logging.info(f"Loading BLIP-2 model ({model_name})...")
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    model.to(device)
    model.eval()

    results = []
    # If partial output exists, load it to skip already processed
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                existing = json.load(f)
                processed_qids = {item["question_id"] for item in existing}
                results = existing
                logging.info(f"Loaded {len(processed_qids)} existing predictions, will skip them.")
        except Exception as e:
            logging.warning(f"Could not read existing output file: {e}")
            processed_qids = set()
    else:
        processed_qids = set()

    # Iterate over videos
    for vid, q_list in tqdm(q_by_video.items(), desc="Videos"):
        video_file = os.path.join(args.video_dir, vid + ".mp4")
        if not os.path.isfile(video_file):
            logging.warning(f"Video file not found: {video_file}. Skipping associated questions.")
            continue
        # Extract frames
        frames = extract_frames(video_file, args.frames)
        if not frames:
            logging.warning(f"No frames extracted for video {vid}. Skipping its questions.")
            continue
        # For each question of this video
        for q in q_list:
            qid = q.get("question_id") or q.get("questionId") or q.get("id")
            if qid in processed_qids:
                continue
            question_text = q.get("question", "")
            options = q.get("options") or q.get("answer_choices") or q.get("choices") or []
            # Expect at least 2 options
            if not options or len(options) < 2:
                logging.warning(f"Question {qid} has insufficient options. Skipping.")
                continue
            answers_per_frame = []
            letters = ['A', 'B', 'C', 'D', 'E']
            for frame in frames:
                prompt = f"Question: {question_text} Options: "
                # Format options as A., B., C.
                for i, opt in enumerate(options[:len(letters)]):
                    prompt += f"{letters[i]}. {opt} "
                prompt += "Answer:"
                inputs = processor(images=frame, text=prompt, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_new_tokens=5)
                ans_text = processor.decode(outputs[0], skip_special_tokens=True).strip()
                answers_per_frame.append(ans_text)
            # Aggregate frame answers by simple majority or pattern
            answer_choice = None
            # If any frame answer starts with a valid letter
            for ans in answers_per_frame:
                if ans and ans[0] in ['A', 'B', 'C']:
                    answer_choice = ans[0]
                    break
            # Try matching full text to option
            if not answer_choice:
                for ans in answers_per_frame:
                    for i, opt in enumerate(options[:len(letters)]):
                        if ans.lower().strip() == opt.lower().strip():
                            answer_choice = letters[i]
                            break
                    if answer_choice:
                        break
            # Fallback: use mode's first character
            if not answer_choice:
                if answers_per_frame:
                    mode_ans = max(set(answers_per_frame), key=answers_per_frame.count)
                    if mode_ans and mode_ans[0] in ['A','B','C']:
                        answer_choice = mode_ans[0]
            # Default in extreme case
            if not answer_choice:
                answer_choice = 'A'
                logging.debug(f"Could not resolve answer for Q{qid}, defaulting to 'A'.")
            results.append({"question_id": qid, "answer": answer_choice})
            processed_qids.add(qid)
            # Save partial output
            with open(args.output, 'w') as f:
                json.dump(results, f)

    logging.info(f"Processed all questions. Total predictions: {len(results)}")
    # Final save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved submission file: {args.output}")


if __name__ == "__main__":
    main()