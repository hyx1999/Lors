from human_eval.data import write_jsonl, read_problems
from fire import Fire
from tqdm import trange, tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import re

ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""

def model_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_text: str,
    model_type: str,
    max_source_length: str = 768,
    max_target_length: str = 256,
):
    inputs = tokenizer(
        input_text + " ",
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True,
        return_token_type_ids=False,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_target_length,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.95,
            temperature=0.8,
        )
    pred_text = tokenizer.decode(
        outputs.sequences[0][len(inputs["input_ids"][0]) :],
        skip_special_tokens=True,
    )
    return pred_text

def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r'^( *)', line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except:
        print("No def line found")
        print(text)
        def_line_space = 0
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)

def generate_one_completion(model, tokenizer, model_type, prompt, template=True):
    if template:
        prompt_in = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt)
    pred_text = model_inference(model, tokenizer, prompt_in, model_type, max_target_length=512)
    post_pred = post_process(pred_text)
    return post_pred

def eval_humaneval(args, model, tokenizer):
    problems = read_problems()
    model_type = "CausalLM"
    
    num_samples_per_task = 5
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(model, tokenizer, model_type, problems[task_id]["prompt"]))
        for task_id in tqdm(problems, desc="Tasks")
        for _ in range(num_samples_per_task)
    ]
    target_name = f"{args.model_name.replace('/', '_')}_humaneval_samples.jsonl"
    write_jsonl(target_name, samples)
    