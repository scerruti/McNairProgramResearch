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

# To determine which option image shows the LEGO object rotated by 90 degrees from the reference image, we can use the holistic VQA approach.

# Create the full question to analyze the image contextually.
full_question = "Which image in the following options shows the LEGO object in the reference image rotated clockwise by 90 degrees? Options: A. Image 2 B. Image 3 C. Image 4 D. Image 5"

# Use the VQA function to determine the correct rotated image.
answer = vqa(image, full_question, None)

# Match the VQA response to the corresponding option.
if 'a' in answer.lower():
    final_result = 'A'
elif 'b' in answer.lower():
    final_result = 'B'
elif 'c' in answer.lower():
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

with open(r"/home/waleedalghaithi/results/lego_2026-02-16_21-04-04/program_execution/image_285_question_285/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        