import os, random
import time

import numpy as np
import torch
import tqdm

from dataloader import get_dataloader
from nets import nn
from utils import util

class PersistentDataLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.iterator = iter(self.dataloader)

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                return next(self.iterator)

def set_global_seed(seed: int, deterministic_kernels: bool = True):
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU/CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / kernel determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if deterministic_kernels:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"Warning: deterministic algorithms not enforced: {e}")

def seed_worker(worker_id: int):
    # Torch gives each worker an initial seed; use it to seed numpy/python
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def __init__(self, args, params, data_path=None, validation_mode=True, trainer_mode=True):
        self.args = args
        self.params = params

        # --- Determinism ---
        if not hasattr(self.args, "seed"):
            self.args.seed = 42  # default
        set_global_seed(self.args.seed, deterministic_kernels=True)
        # single RNG passed to DataLoader/Sampler to make shuffling reproducible
        self._dl_generator = torch.Generator(device="cpu").manual_seed(self.args.seed)


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)

        self.model = nn.yolo_v8_n(len(params["names"].values())).to(self.device)
        self.ema = util.EMA(self.model)

        
        if not data_path:
            data_path = os.getenv("DATA_PATH")
        print("data_path: ", data_path)
        torch.multiprocessing.set_start_method("spawn", force=True)
        #self.train_loader = get_dataloader(data_path, "train", args, params)
        if trainer_mode:
            self.optimizer = self.configure_optimizer()
            self.scheduler = self.configure_scheduler()
            self.train_iter = None
            self.train_loader = PersistentDataLoader(
                get_dataloader(
                    data_path,
                    "train",
                    self.args,
                    self.params,
                    worker_init_fn=seed_worker,     
                    generator=self._dl_generator,
                    shuffle=True
                    )
                )            
            self.usage_counter = np.zeros(len(self.train_loader.dataloader.dataset), dtype=int)
            self.prev_iterations = 0
            self.round_index = 0
        if validation_mode:
            self.val_loader = get_dataloader(
                data_path,
                "valid",
                self.args,
                self.params,
                num_workers=8,
                worker_init_fn=seed_worker,          
                generator=self._dl_generator,
                shuffle=False
            )

        

    def configure_optimizer(self):
        accumulate = max(round(64 / self.args.batch_size), 1)
        self.params["weight_decay"] *= self.args.batch_size * accumulate / 64
        p = [], [], []
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                p[2].append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):
                p[1].append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                p[0].append(v.weight)

        optimizer = torch.optim.Adam(p[2], self.params["lr0"])
        optimizer.add_param_group(
            {"params": p[0], "weight_decay": self.params["weight_decay"]}
        )
        optimizer.add_param_group({"params": p[1]})
        return optimizer

    def configure_scheduler(self):
        
        def lr(x):
            return  (1 - min(x / self.args.epochs, 1)) * (1.0 - self.params["lrf"]) + self.params[
                "lrf"
            ]

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr, last_epoch=-1)

    def lr_schedule(self, roundindex):
        """Learning rate decay schedule."""
        
        return (1 - min(roundindex / self.args.epochs, 1)) * (1.0 - self.params["lrf"]) + self.params[
            "lrf"
        ]
    
    

    def train(self):
        device = self.device

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("trainer train starts")
        t0 = time.time()

        accumulate = max(round(64 / self.args.batch_size), 1)
        
        # Use autocast + GradScaler only if CUDA is available
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            autocast = torch.cuda.amp.autocast
        else:
            scaler = None
            # no-op context manager
            from contextlib import nullcontext
            autocast = nullcontext

        criterion = util.ComputeLoss(self.model, self.params)
        num_warmup = 1000

        self.model.train()
        warm_up = False

        print(("\n" + "%10s" * 5) % ("updates", "memory", "warm_up", "x", "loss"))

        p_bar = tqdm.tqdm(range(self.args.local_updates))

        self.optimizer.zero_grad()
        m_loss = util.AverageMeter()

        for _ in p_bar:
            samples, targets, _, indices = next(self.train_loader)
            self.usage_counter[indices] += 1

            x = self.prev_iterations

            # ✅ Move to correct device
            samples = samples.to(device).float() / 255
            targets = targets.to(device)

            # Warmup logic (unchanged)
            if x <= num_warmup:
                warm_up = True
                xp = [0, num_warmup]
                fp = [1, 64 / self.args.batch_size]
                accumulate = max(1, np.interp(x, xp, fp).round())
                for j, y in enumerate(self.optimizer.param_groups):
                    y.setdefault("initial_lr", y["lr"])
                    if j == 0:
                        fp = [self.params["warmup_bias_lr"], y["initial_lr"] * self.lr_schedule(self.round_index)]
                    else:
                        fp = [0.0, y["initial_lr"] * self.lr_schedule(self.round_index)]
                    y["lr"] = np.interp(x, xp, fp)
                    if "momentum" in y:
                        fp = [self.params["warmup_momentum"], self.params["momentum"]]
                        y["momentum"] = np.interp(x, xp, fp)

            # Forward pass
            with autocast():
                outputs = self.model(samples)
                loss = criterion(outputs, targets)

            m_loss.update(loss.item(), samples.size(0))
            loss *= self.args.batch_size

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if x % accumulate == 0:
                if scaler:
                    scaler.unscale_(self.optimizer)
                    util.clip_gradients(self.model)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    util.clip_gradients(self.model)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)

            # Progress bar
            if device.type == "cuda":
                memory = f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
            else:
                memory = "CPU"

            p_bar.set_description(
                ("%10s" * 3 + "%10.4g" * 2)
                % (
                    f"{self.prev_iterations%self.args.local_updates + 1}/{self.args.local_updates}",
                    memory,
                    str(warm_up),
                    x,
                    m_loss.avg,
                )
            )

            self.prev_iterations += 1

        self.scheduler.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        self.round_index += 1
        print("training done")
        t1 = time.time()
        print("time: ", t1 - t0)

    

    @torch.no_grad()
    def validate(self, val_model, threshold=0.5):
        device = self.device
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("validate start")
        t0 = time.time()

        val_model.eval()

        # Configure
        iou_v = torch.linspace(0.5, 0.95, 10).to(device)  # IoU thresholds
        n_iou = iou_v.numel()

        m_pre, m_rec, map50, mean_ap = 0.0, 0.0, 0.0, 0.0
        metrics = []

        p_bar = tqdm.tqdm(self.val_loader, desc=("%10s" * 3) % ("precision", "recall", "mAP"))

        for samples, targets, shapes, _ in p_bar:
            # ✅ Move data to device
            samples = samples.to(device).float() / 255
            targets = targets.to(device)

            _, _, height, width = samples.shape

            # Inference
            outputs = val_model(samples)

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
            outputs = util.non_max_suppression(outputs, threshold, 0.65)

            # Metrics
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

                if samples[i] is None:
                    print("samples[i]: ", samples[i])
                    continue

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, *torch.zeros((3, 0), device=device)))
                    continue

                if shapes[i] is None:
                    print("shapes[i] is None: ", shapes[i])
                    continue

                detections = output.clone()
                util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

                # Evaluate
                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()
                    tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2
                    tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2
                    tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2
                    tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2
                    util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                    correct = np.zeros((detections.shape[0], iou_v.shape[0]), dtype=bool)

                    t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                    iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]
                    for j in range(len(iou_v)):
                        x = torch.where((iou >= iou_v[j]) & correct_class)
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                            matches = matches.cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            correct[matches[:, 1].astype(int), j] = True
                    correct = torch.tensor(correct, dtype=torch.bool, device=device)

                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

        # Compute metrics
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

        # Print results
        print("%10.3g" * 3 % (m_pre, m_rec, mean_ap))

        # Reset model to float32 (important if training continues)
        val_model.float()

        print("validation done")
        t1 = time.time()
        print("validation time: ", t1 - t0)
        return m_pre, m_rec, map50, mean_ap