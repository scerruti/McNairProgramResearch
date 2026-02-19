SIGNATURE_PROMPT = """
Propose only new method signatures to add to the existing API.

Available Primitives: image, float, string, list, int

Current API:
{signatures}

Next, I will ask you a series of questions that reference an image and are solvable with a python program that uses the API I have provided so far. Please propose new method signatures with associated docstrings to add to the API that would help modularize the programs that answer the questions. 

For each proposed method, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>. Do not propose methods that are already in the API.

Please ensure that you ONLY add new methods when necessary. Do not add new methods if you can solve the problem with combinations of the previous methods!

Do not assume units! Some questions will ask for centimeters, some for meters. Adapt accordingly

Added methods should be simple, building minorly on the methods that already exist. Do NOT assume that the objects you see in these questions are all of the objects you will see, keep the methods general.

Here are some helpful tips and definitions:
1) 2D distance/size refers to distance/size in pixel space.
2) 3D distance/size refers to distance/size in the real world. 3D size is equal to 2D size times the depth of the object.
3) We define (width, height, length) as the values along the (x, y, z) axis. Width = x axis, height = y axis, length = z axis.
4) "Depth" measures distance from the camera in 3D.
5) Some questions may present hypothesis measurements (e.g. "X is 3.5m wide"), these are hypothesis measurements and should be used ONLY to scale your outputs accordingly.
6) Do NOT round your answers! Always leave your answers as decimals even when it feels intuitive to round or ceiling your answer - do not do it!
7) When a query asks to find all objects in a container just count the number of objects.

Importantly, new methods MUST start with an underscore. As an example, you may define a "_get_material" method. Please ensure you ALWAYS start the name with an underscore.

Again, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>.

DO NOT INCLUDE ``` tags!

Here is the question:
{question}
"""

SIGNATURE_PROMPT_LEGO = """
Propose only new method signatures to add to the existing API.

Available Primitives: image, float, string, list, int, bool

Current API:
{signatures}

Next, I will ask you a series of questions about LEGO assembly images. These questions test spatial reasoning about LEGO brick constructions, including:
- Adjacency: which bricks are next to each other
- Height: comparing heights of LEGO structures
- Position: where bricks are located relative to each other
- Rotation: identifying rotated configurations
- Ordering: determining the sequence of assembly steps
- Next step: predicting the next brick placement
- Multi-view: reasoning about 3D structures from multiple viewpoints
- Dependency: understanding which bricks depend on others for support
- Outlier: identifying which assembly doesn't belong

Please propose new method signatures with associated docstrings to add to the API that would help modularize the programs that answer these LEGO spatial reasoning questions.

For each proposed method, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>. Do not propose methods that are already in the API.

Please ensure that you ONLY add new methods when necessary. Do not add new methods if you can solve the problem with combinations of the previous methods!

Added methods should be simple, building minorly on the methods that already exist. Do NOT assume that the objects you see in these questions are all of the objects you will see, keep the methods general.

Here are some helpful tips:
1) LEGO images show brick assemblies from specific viewpoints. Use visual question answering to determine brick colors, positions, and relationships.
2) Bricks are stacked in layers. "Height" usually refers to the number of layers or vertical extent.
3) "Adjacent" means bricks are directly next to each other (touching sides).
4) For ordering/sequence questions, you need to reason about which bricks were placed first based on the structure.
5) For HEIGHT comparison questions (higher/shorter/taller in 3D space), do NOT create any new height comparison methods. Height questions should be answered using holistic VQA directly — the images already have colored markers visible to GPT-4o.

Importantly, new methods MUST start with an underscore. As an example, you may define a "_get_brick_color" method. Please ensure you ALWAYS start the name with an underscore.

Again, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>.

DO NOT INCLUDE ``` tags!

Here is the question:
{question}
"""

SIGNATURE_PROMPT_CLEVR = """
Propose only new method signatures to add to the existing API.

Available Primitives: image, int, string, list

Current API:
{signatures}

Next, I will ask you a series of questions that reference an image and are solvable with a python program that uses the API I have provided so far. Please propose new method signatures with associated docstrings to add to the API that would help modularize the programs that answer the questions. 

For each proposed method, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>. Do not propose methods that are already in the API.

Please ensure that you ONLY add new methods when necessary. Do not add new methods if you can solve the problem with combinations of the previous methods!

Your methods should take as parameters and return ONLY the primitives given above.

Added methods should be simple, building minorly on the methods that already exist. Do NOT assume that the objects you see in these questions are all of the objects you will see, keep the methods general.

Importantly, new methods MUST start with an underscore. As an example, you may define a "_get_material" method. Please ensure you ALWAYS start the name with an underscore.

Again, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>.

DO NOT INCLUDE ``` tags!

{question}
"""
