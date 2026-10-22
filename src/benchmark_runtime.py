"""Synthetic resource smoke test only. Does not train a scientific model.

Run in a fresh process. Reports whole-process peak RSS, not free system memory.
"""
import argparse
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--steps', type=int, default=10)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output filename to preserve previous records')
    if a.size < 32 or a.size % 8 or min(a.batch_size, a.steps) < 1:
        p.error('size must be >=32 and divisible by 8; batch/steps positive')
    record = dict(scope='synthetic resource smoke test; not MRI results',
                  clinical_data_used=False, scientific_models_trained=0,
                  platform=platform.platform(), machine=platform.machine(),
                  python=platform.python_version(), size=a.size,
                  batch_size=a.batch_size, steps=a.steps)
    try:
        import torch
        from monai_components import model
        from monai.losses import DiceCELoss
        record['packages'] = {x: importlib.metadata.version(x)
                              for x in ('torch', 'monai', 'numpy')}
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        record['device'] = str(device)
        torch.manual_seed(61900)
        torch.set_num_threads(2)
        net = model().to(device)
        record['parameters'] = sum(x.numel() for x in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        loss_fn = DiceCELoss(sigmoid=True)
        x = torch.randn(a.batch_size, 1, a.size, a.size, device=device)
        y = (torch.rand_like(x) > .8).float()

        def sync():
            if device.type == 'mps':
                torch.mps.synchronize()

        def step():
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all()
                   for v in net.parameters()):
                raise RuntimeError('Nonfinite gradient')
            opt.step()
            return float(loss.detach())

        step()  # Untimed warm-up includes optimizer allocation.
        sync()
        times = []
        for _ in range(a.steps):
            start = time.perf_counter()
            last_loss = step()
            sync()
            times.append(time.perf_counter() - start)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record.update(status='passed', last_synthetic_loss=last_loss,
                      seconds_per_step=times,
                      mean_seconds_per_step=sum(times)/len(times),
                      peak_process_rss_bytes=peak if platform.system() == 'Darwin' else peak*1024,
                      limitation='Synthetic compute only; no real epoch or volume-inference benchmark')
        if device.type == 'mps':
            record['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
            record['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
    except Exception as error:
        record.update(status='blocked_or_failed', error=repr(error))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps(record, indent=2))
    if record['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
