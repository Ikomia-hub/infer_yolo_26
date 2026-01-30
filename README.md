<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_yolo_26</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_yolo_26">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_yolo_26">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_yolo_26/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_yolo_26.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

YOLO26 object detection inference powered by Ultralytics models.

![illustration instance segmentation](https://raw.githubusercontent.com/Ikomia-hub/infer_yolo_26_seg/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_26", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_bike_rider.jpeg?raw=true")
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

Parameters:
- `model_name`: YOLO26 model variant (`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`).
- `cuda`: Enable CUDA if available (`True`/`False`).
- `input_size`: Inference resolution (int, e.g. `640`).
- `conf_thres`: Confidence threshold (float 0-1).
- `iou_thres`: IoU threshold for NMS (float 0-1).
- `model_weight_file`: Custom `.pt` path (empty to use default weights).


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_26", auto_connect=True)

algo.set_parameters({
    "model_name": "yolo26m",
    "cuda": "True",
    "input_size": "640",
    "conf_thres": "0.25",
    "iou_thres": "0.7",
    "model_weight_file": ""
})

# Run on your image  
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_bike_rider.jpeg?raw=true")
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_26", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_bike_rider.jpeg?raw=true")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
