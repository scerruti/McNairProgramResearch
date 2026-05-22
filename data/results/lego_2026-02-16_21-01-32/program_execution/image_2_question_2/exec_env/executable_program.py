import math

def _compare_3D_heights_shorter(image, bbox1, bbox2):

    
    comparison_result = compare_3D_heights(image, bbox1, bbox2)
    
    if comparison_result == 'first':
        return 'second'
    elif comparison_result == 'second':
        return 'first'
    else:
        return 'same'
    

# PROGRAM STARTS HERE

full_question = "Which LEGO object is shorter in 3D space? Options: A. The LEGO piece pointed by the red arrow. B. The LEGO piece pointed by the blue arrow. C. They are the same height. Answer with only the letter of the correct option."
answer = vqa(image, full_question, None)
if 'a' in answer.lower():
    final_result = 'A'
elif 'b' in answer.lower():
    final_result = 'B'
else:
    final_result = 'C'


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"/home/waleedalghaithi/results/lego_2026-02-16_21-01-32/program_execution/image_2_question_2/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        