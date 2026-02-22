import math
def loc(image, object_prompt):

	return [[25, 25, 50, 50]]

import math
def vqa(image, question, bbox):

	return ""

import math
def same_object(image, bbox1, bbox2):

	return False

import math
def depth(image, bbox):

	return 1.0

import math
def compare_3D_heights(image, bbox1, bbox2):

	return ""

import math
def get_2D_object_size(image, bbox):

	return (50, 50)


# PROGRAM STARTS HERE

# Locate the objects in the image based on the provided arrow descriptions
bbox1 = loc(image, arrow1)[0]
bbox2 = loc(image, arrow2)[0]

# Compare the 3D heights of the objects located by the arrows
result = compare_3D_heights(image, bbox1, bbox2)

# Return the result of the comparison
final_result = result


# WRITE NAMESPACE
import json
def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"/home/waleedalghaithi/results/lego_2026-02-16_21-04-04/api_generator/_compare_height_with_arrows/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        