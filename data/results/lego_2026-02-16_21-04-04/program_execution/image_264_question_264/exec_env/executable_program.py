import math

def _compare_height_with_arrows(image, arrow1, arrow2):

    
    # Locate the objects in the image based on the provided arrow descriptions
    bbox1 = loc(image, arrow1)[0]
    bbox2 = loc(image, arrow2)[0]
    
    # Compare the 3D heights of the objects located by the arrows
    result = compare_3D_heights(image, bbox1, bbox2)
    
    # Return the result of the comparison
    return result
    

def _determine_installation_point(current_state, piece_to_install, after_installation, options):

    # Iterate over each option and check if applying the piece results in the after_installation state
    for i, option in enumerate(options):
        # Identify the current bounding boxes of the piece in the current and after installation states
        piece_bbox = loc(piece_to_install, "piece to install")
        current_bbox = loc(current_state, "current state")
        after_bbox = loc(after_installation, "after installation state")
        
        # Check if placing the piece at this option results in the after installation state
        if same_object(after_installation, after_bbox[0], piece_bbox[0]) and same_object(current_state, current_bbox[0], option):
            return chr(65 + i)  # Return 'A', 'B', 'C', or 'D' based on the index

    return None


def _is_rotation_necessary(base_structure, final_structure, additional_piece):

    # Locate the additional piece in the base structure
    base_piece_bboxes = loc(base_structure, "bricks")
    additional_piece_bbox = loc(additional_piece, "top brick")[0]

    # Check if the additional piece is already in the correct orientation in the base structure
    for bbox in base_piece_bboxes:
        if same_object(base_structure, bbox, additional_piece_bbox):
            return False

    # If not found in the original orientation, check in the final structure
    final_piece_bboxes = loc(final_structure, "bricks")
    for bbox in final_piece_bboxes:
        if same_object(final_structure, bbox, additional_piece_bbox):
            return True

    return False


def _check_adjacency(image, arrow1, arrow2):

    
    # Locate the objects pointed by the arrows
    bbox1 = loc(image, arrow1)[0]
    bbox2 = loc(image, arrow2)[0]
    
    # Use the VQA to check if the objects are adjoining
    result = vqa(image=image, question='Are these two LEGO objects directly touching or adjoining?', bbox=bbox1)
    
    # Return True if the result indicates they are adjoining, otherwise False
    return 'yes' in result.lower()
    

def _identify_incorrect_step(current_state, target_state, steps):

    # Compare each step with the target state to find the incorrect one
    for i, step in enumerate(steps):
        # Assume that a step is correct if it results in the target state
        if vqa(target_state, 'Is this the correct state after this step?', step) == 'yes':
            continue
        else:
            # Return the incorrect step's corresponding letter
            return chr(65 + i)  # 65 is the ASCII value for 'A'

    # In case all steps seem correct, which shouldn't happen, return an empty string
    return ''


# PROGRAM STARTS HERE

# The task is to determine which optional image shows the LEGO object rotated 60 degrees clockwise
# from the reference image. We will use the API to analyze the images and determine the correct option.

# The images are identified as follows:
# Reference image (x_0): <image 1>
# Option A: <image 2>
# Option B: <image 3>
# Option C: <image 4>
# Option D: <image 5>

# We will use the VQA function to compare the reference image with each option image
# to determine which one matches a 60-degree rotation.

# Define the question regarding the rotation comparison
rotation_question = "Does this image show the LEGO object rotated 60 degrees clockwise relative to the reference image?"

# Check each option
is_a_correct = vqa(image, rotation_question, bbox=None)  # Option A
is_b_correct = vqa(image, rotation_question, bbox=None)  # Option B
is_c_correct = vqa(image, rotation_question, bbox=None)  # Option C
is_d_correct = vqa(image, rotation_question, bbox=None)  # Option D

# Determine which option matches the 60-degree rotation
if 'yes' in is_a_correct.lower():
    final_result = 'A'
elif 'yes' in is_b_correct.lower():
    final_result = 'B'
elif 'yes' in is_c_correct.lower():
    final_result = 'C'
else:
    final_result = 'D'


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"/home/waleedalghaithi/results/lego_2026-02-16_21-04-04/program_execution/image_264_question_264/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        