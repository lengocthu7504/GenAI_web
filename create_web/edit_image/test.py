

def check(task,prompt, negative_prompt):
    tasks = [
{
    "task": "object-removal",
    "guidance_scale": 12,
    "prompt": "",
    "negative_prompt": "",
},
{
    "task": "shape-guided",
    "guidance_scale": 7.5,
    "prompt": prompt,
    "negative_prompt": negative_prompt,
},
{
    "task": "inpaint",
    "guidance_scale": 7.5,
    "prompt": prompt,
    "negative_prompt": negative_prompt,
},
{
    "task": "image-outpainting",
    "guidance_scale": 7.5,
    "prompt": prompt,
    "negative_prompt": negative_prompt,
},
]
    for i in tasks:
        if i["task"] == task:
            return print("task ",i["task"]," guidance_scale","   prompt ",i["prompt"],"negative_prompt ",i["negative_prompt"])
prompt = "11"
negative_prompt ="-11"
task = "image-outpainting"        
check(task,prompt,negative_prompt)