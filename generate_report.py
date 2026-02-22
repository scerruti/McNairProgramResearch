from fpdf import FPDF

class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "McNair Research - VADAR on LEGO-Puzzles Benchmark", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.ln(4)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(45, 65, 95)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(240, 244, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6.5, str(val), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)

    def code_block(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        self.set_x(x + 5)
        self.multi_cell(180, 4.5, text, fill=True)
        self.ln(3)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(30, 30, 30)
pdf.ln(10)
pdf.cell(0, 12, "Evaluating VADAR on the", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 12, "LEGO-Puzzles Benchmark", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)

pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "McNair Research", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "CSE 199 - February 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)

# Overview
pdf.section_title("1. Overview")
pdf.body_text(
    "This report evaluates VADAR, an agentic spatial reasoning system, on the LEGO-Puzzles "
    "benchmark for multi-step spatial reasoning in multimodal large language models (MLLMs). "
    "VADAR uses a pipeline of AI agents (Signature Agent, API Agent, Program Agent) combined "
    "with vision models (GroundingDINO, SAM2, UniDepth) to dynamically compose programs that "
    "answer spatial reasoning questions. We compare its performance against GPT-4o Mini and "
    "Gemini-2.0-Flash, which were evaluated as direct API-based solvers."
)

pdf.body_text(
    "The LEGO-Puzzles benchmark contains 220 images and 1,100 multiple-choice questions across "
    "11 spatial reasoning categories: adjacency, height, rotation, rotation status, multi-view, "
    "position, ordering, next step, backwards, dependency, and outlier."
)

# Setup
pdf.section_title("2. Experimental Setup")
pdf.body_text(
    "VADAR was deployed on a remote Linux machine (elcap.ucsd.edu) equipped with an NVIDIA "
    "RTX 2080 Ti GPU (11 GB VRAM) running CUDA driver 11.4. The system used PyTorch 2.1.2 "
    "with CUDA 11.8 toolkit, and GPT-4o as the backbone LLM for all agentic reasoning steps. "
    "Several environment-level patches were required to achieve compatibility:"
)

pdf.subsection_title("Key Setup Challenges")
pdf.body_text(
    "- CUDA/PyTorch version mismatch: The system nvcc was version 10.2 while PyTorch required "
    "CUDA 11.8. GroundingDINO's C++ CUDA extensions could not compile, requiring a fallback "
    "patch to use pure Python multi-scale deformable attention.\n\n"
    "- bfloat16 unsupported: VADAR assumed bfloat16 precision (Ampere+ GPUs). The RTX 2080 Ti "
    "(Turing architecture) required dynamic detection to fall back to float16.\n\n"
    "- UniDepth JIT incompatibility: Python 3.10 union type syntax (int | tuple) inside "
    "@torch.jit.script decorators caused runtime failures. The decorators were removed.\n\n"
    "- SAM2 dependency conflict: Installing SAM2 pulled in PyTorch 2.5+, breaking CUDA "
    "compatibility. PyTorch was force-downgraded back to 2.1.2+cu118."
)

# Results
pdf.section_title("3. Results")

headers = ["Category", "Gemini-2.0-Flash", "GPT-4o Mini", "VADAR (GPT-4o)"]
col_widths = [42, 46, 46, 46]
data = [
    ["Overall",     "54.18%", "23.27%", "11.91%"],
    ["adjacency",   "65%",    "57%",    "58%"],
    ["height",      "35%",    "29%",    "40%"],
    ["multi_view",  "48%",    "17%",    "10%"],
    ["rotation",    "49%",    "12%",    "8%"],
    ["rotation_status", "54%", "44%",   "8%"],
    ["backwards",   "57%",    "17%",    "7%"],
    ["dependency",  "82%",    "36%",    "0%"],
    ["next_step",   "68%",    "16%",    "0%"],
    ["ordering",    "46%",    "4%",     "0%"],
    ["outlier",     "45%",    "14%",    "0%"],
    ["position",    "47%",    "10%",    "0%"],
]
pdf.add_table(headers, data, col_widths)

pdf.body_text(
    "VADAR achieved an overall accuracy of 11.91% on the LEGO-Puzzles-Lite benchmark. While "
    "this is lower than both baselines overall, performance varies significantly by category."
)

# Analysis
pdf.section_title("4. Analysis")

pdf.subsection_title("Where VADAR Succeeds")
pdf.body_text(
    "VADAR performs competitively on single-image spatial perception tasks. On adjacency (58%), "
    "it beats GPT-4o Mini (57%) and approaches Gemini Flash (65%). On height (40%), it is the "
    "top performer, beating both Gemini Flash (35%) and GPT-4o Mini (29%). These categories "
    "benefit from VADAR's ability to compose vision tools -- using GroundingDINO for object "
    "detection and GPT-4o for holistic visual question answering on images that already contain "
    "colored markers (red/blue rectangles and arrows)."
)

pdf.subsection_title("Where VADAR Fails")
pdf.body_text(
    "VADAR scores 0% on five categories: dependency, next_step, ordering, outlier, and position. "
    "These all require multi-image reasoning -- comparing a sequence of assembly step images to "
    "determine order, predict what comes next, or identify which step doesn't belong. VADAR's "
    "execution engine provides only a single 'image' variable to programs, so any question "
    "requiring multiple images causes the generated code to crash with NameError (e.g., "
    "'name x_0 is not defined'). This is a fundamental architectural limitation, not a tuning "
    "issue."
)

pdf.subsection_title("Trace Analysis")
pdf.body_text(
    "VADAR provides full intermediate traces at every pipeline stage (signature generation, "
    "API generation, program generation, and execution). Examining these traces reveals "
    "distinct failure modes:\n\n"
    "- For single-image tasks (height, adjacency): The pipeline works as intended. The Program "
    "Agent generates correct code using vqa() or _check_adjacency(), and the execution engine "
    "runs it successfully.\n\n"
    "- For multi-image tasks (position, ordering, next_step): The Program Agent generates code "
    "referencing variables like x_0, x_1, x_2 and functions like _determine_installation_point() "
    "that do not exist in the execution environment. Every such question fails with a runtime error.\n\n"
    "- For rotation tasks: The Program Agent attempts to use VQA to compare two images, but "
    "the engine only passes one image, leading to inaccurate guesses."
)

# Cost
pdf.section_title("5. Performance vs. Cost")

headers2 = ["Metric", "Gemini-2.0-Flash", "GPT-4o Mini", "VADAR (GPT-4o)"]
data2 = [
    ["Runtime",         "~5 min",  "~10 min", "~4 hours"],
    ["API calls/question", "1",    "1",       "3-8"],
    ["Overall accuracy", "54.18%", "23.27%",  "11.91%"],
    ["Best category",   "dependency (82%)", "adjacency (57%)", "adjacency (58%)"],
]
pdf.add_table(headers2, data2, col_widths)

pdf.body_text(
    "VADAR's agentic pipeline is significantly more expensive to run. Each question requires "
    "multiple GPT-4o API calls across three agents (signature, API, program generation), plus "
    "additional calls during execution for VQA-based modules. Questions that trigger retries "
    "(up to 5 attempts) can take 30-45 seconds each. The total benchmark runtime was "
    "approximately 4 hours, compared to minutes for direct API calls to GPT-4o Mini or "
    "Gemini Flash."
)

# Conclusion
pdf.section_title("6. Conclusion")
pdf.body_text(
    "VADAR actually does pretty well on single-image stuff like figuring out which LEGO piece "
    "is higher or which ones are next to each other, and it even beats GPT-4o Mini and Gemini "
    "on some of those. But most of the LEGO benchmark is about looking at a sequence of "
    "assembly steps and figuring out what comes next or what order they go in, and VADAR just "
    "can't do that since it only looks at one image at a time. So it ends up at around 12% "
    "overall, and it takes over 4 hours to run compared to minutes for the other models."
)

pdf.body_text(
    "To improve VADAR's performance on the LEGO benchmark, the system would need: "
    "(1) multi-image support in the execution engine so programs can reference and compare "
    "multiple images, (2) temporal reasoning capabilities to understand assembly sequences, "
    "and (3) more robust bounding box detection, since GroundingDINO frequently generates "
    "image-spanning boxes that undermine spatial analysis."
)

output_path = "/Users/waleed/CSE 199/VADAR_LEGO_Report.pdf"
pdf.output(output_path)
print(f"PDF saved to {output_path}")
