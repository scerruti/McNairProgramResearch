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

# Since there have been issues with undefined images, let's address the problem by assuming that
# the question and images are represented in a way that allows us to use the API effectively.

# The main image is assumed to be initialized in 'image', and we will use a holistic approach to determine the answer.

# Define the full question with options
full_question = """You are a specialized LEGO 3D rotation analyzer. You will be provided with one reference image (x_0) and four optional images (A, B, C, D). 
Each optional image shows the same LEGO object, rotated clockwise from a top-down perspective (looking down on the LEGO from above) around its center by one of the following angles: 30°, 60°, 90°, or 120°. 
The images are presented in a random order, so it is not specified which image corresponds to which rotation angle. 
Your task is to determine which optional image matches the specified rotation relative to the reference image. 
Your answers should be based solely on the provided LEGO 3D data, without any additional assumptions. 
Keep your responses clear, direct, and focused on the question. 
Please respond with only the letter corresponding to your choice (A, B, C, or D).
Which image in the following options shows the LEGO object in the reference image rotated clockwise by 90 degrees?
Options: A. <image 2> B. <image 3> C. <image 4> D. <image 5>"""

# Use VQA holistically to determine the option that matches the 90-degree rotation
answer = vqa(image, full_question, None)

# Map the answer to the correct option letter
if 'a' in answer.lower():
    final_result = 'A'
elif 'b' in answer.lower():
    final_result = 'B'
elif 'c' in answer.lower():
    final_result = 'C'
elif 'd' in answer.lower():
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

with open(r"/home/waleedalghaithi/results/lego_2026-02-16_21-04-04/program_execution/image_269_question_269/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        