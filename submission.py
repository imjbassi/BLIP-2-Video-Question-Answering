import json

with open("submission.json") as f:
    nested_submission = json.load(f)

flat_dict = {}
for video_id, answers in nested_submission.items():
    for i, answer in enumerate(answers):
        flat_dict[f"{video_id}_q{i}"] = answer

with open("submission_fixed.json", "w") as f:
    json.dump(flat_dict, f, indent=2)

print("✅ submission_fixed.json created")
