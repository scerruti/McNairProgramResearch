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

# Since the images are not defined with specific variable names, I will assume that the image variable
# named "image" is already initialized and that the various components of the problem need to be extracted 
# from this "image" using the provided API. 

# Using the _determine_installation_point function requires the current state, the piece to install, 
# the final target state, and the options. 

# Let's assume the images are part of a larger image and need to be extracted using bounding boxes or some mechanism.
# For this example, we'll assume these images are parts of the single initialized "image" variable.

# Define bounding boxes or some way to extract the respective images from the "image" variable.
# These are placeholders and need to be defined according to how the images are segmented.
current_state_bbox = [0, 0, 100, 100]  # Example bounding box for x_1
piece_to_install_bbox = [100, 0, 200, 100]  # Example bounding box for x_2
after_installation_bbox = [200, 0, 300, 100]  # Example bounding box for x_3
option_A_bbox = [0, 100, 100, 200]  # Example bounding box for option A
option_B_bbox = [100, 100, 200, 200]  # Example bounding box for option B
option_C_bbox = [200, 100, 300, 200]  # Example bounding box for option C
option_D_bbox = [300, 100, 400, 200]  # Example bounding box for option D
option_E_bbox = [400, 100, 500, 200]  # Example bounding box for option E

# Extract images using these bounding boxes (assuming a function or method to do so)
# Here, I am just using placeholders to represent these extracted images.
current_state = image  # Placeholder for extracted image using current_state_bbox
piece_to_install = image  # Placeholder for extracted image using piece_to_install_bbox
after_installation = image  # Placeholder for extracted image using after_installation_bbox
options = [image, image, image, image, image]  # Placeholders for option images using respective bounding boxes

# Determine the correct option using the provided API function
final_result = _determine_installation_point(current_state, piece_to_install, after_installation, options)


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"/home/waleedalghaithi/results/lego_2026-02-16_21-04-04/program_execution/image_651_question_651/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        