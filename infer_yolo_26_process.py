import copy
import os

import torch

from ikomia import core, dataprocess, utils
from ultralytics import YOLO
from ultralytics import download


class InferYolo26Param(core.CWorkflowTaskParam):
    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "yolo26m"
        self.cuda = torch.cuda.is_available()
        self.input_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.7
        self.update = False
        self.model_weight_file = ""

    def set_values(self, param_map):
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.update = True

    def get_values(self):
        param_map = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "input_size": str(self.input_size),
            "conf_thres": str(self.conf_thres),
            "iou_thres": str(self.iou_thres),
            "update": str(self.update),
            "model_weight_file": str(self.model_weight_file)
        }
        return param_map


class InferYolo26(dataprocess.CObjectDetectionTask):
    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)

        if param is None:
            self.set_param_object(InferYolo26Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.repo = "ultralytics/assets"
        self.version = "v8.4.0"
        self.device = torch.device("cpu")
        self.classes = None
        self.model = None
        self.half = False

    def get_progress_steps(self):
        return 1

    def _load_model(self):
        param = self.get_param_object()
        self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        self.half = True if param.cuda and torch.cuda.is_available() else False

        if param.model_weight_file:
            self.model = YOLO(param.model_weight_file)
        else:
            model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
            os.makedirs(model_folder, exist_ok=True)
            model_weights = os.path.join(str(model_folder), f"{param.model_name}.pt")

            if not os.path.isfile(model_weights):
                url = f"https://github.com/{self.repo}/releases/download/{self.version}/{param.model_name}.pt"
                download(url=url, dir=model_folder, unzip=True)

            self.model = YOLO(model_weights)

        param.update = False

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def run(self):
        self.begin_task_run()

        param = self.get_param_object()

        img_input = self.get_input(0)
        src_image = img_input.get_image()

        if param.update:
            self._load_model()

        results = self.model.predict(
            src_image,
            save=False,
            imgsz=param.input_size,
            conf=param.conf_thres,
            iou=param.iou_thres,
            half=self.half,
            device=self.device
        )

        self.classes = list(results[0].names.values())
        self.set_names(self.classes)

        boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        class_idx = results[0].boxes.cls

        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_idx)):
            box = box.detach().cpu().numpy()
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            width = x2 - x1
            height = y2 - y1
            self.add_object(
                i,
                int(cls),
                float(conf),
                float(x1),
                float(y1),
                float(width),
                float(height)
            )

        self.emit_step_progress()
        self.end_task_run()


class InferYolo26Factory(dataprocess.CTaskFactory):
    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        self.info.name = "infer_yolo_26"
        self.info.short_description = "Inference with YOLO26 models (Ultralytics)"
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.min_ikomia_version = "0.16.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2026
        self.info.license = "AGPL-3.0"
        self.info.documentation_link = "https://docs.ultralytics.com/"
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_26"
        self.info.original_repository = "https://github.com/ultralytics/ultralytics"
        self.info.keywords = "YOLO, YOLO26, object, detection, ultralytics, real-time"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        return InferYolo26(self.info.name, param)
