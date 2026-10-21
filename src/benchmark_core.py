"""Model adapters used by the synthetic Mithai experiment. Extracted from the pinned source benchmark_core.py; unrelated evaluation code omitted."""


import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import numpy as np


SEED = 67067


QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


MAX_CONTEXT = 2048


PINS = {
    'yug': dict(repo='birlaailabs/yug', revision='ec5375888b672e334e8a258ce03d7e57fcfbb610',
                package='yug @ git+https://github.com/birla-ai-labs/yug.git@2ad7da7133d5cb5b27be3ac1cc5e7963e2ba4e79'),
    'timesfm-2.5': dict(repo='google/timesfm-2.5-200m-pytorch', revision='1d952420fba87f3c6dee4f240de0f1a0fbc790e3',
                        package='timesfm @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688'),
    'chronos-bolt-base': dict(repo='amazon/chronos-bolt-base', revision='5d9f166d69f47aef3401367a7b842e78fe97b121',
                              package='chronos-forecasting @ git+https://github.com/amazon-science/chronos-forecasting.git@fcd09fe8b6bf57643bde7847bf42a6719b7ee205'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Recorder:
    """Append-only JSONL writer; every experiment row is flushed immediately."""

    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)

    def append(self, name, row):
        with open(self.out / f'{name}.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')

    def write(self, name, value):
        (self.out / name).write_text(json.dumps(value, indent=1, default=str))

    def log(self, msg, **kw):
        row = dict(utc=utc(), msg=msg, **kw)
        print(json.dumps(row), flush=True)
        self.append('stages', row)


def check_forecast(q, horizon):
    q = np.asarray(q, dtype=float)
    if q.shape != (horizon, 9):
        raise RuntimeError(f'SHAPE {q.shape}')
    if not np.isfinite(q).all():
        raise RuntimeError('NONFINITE')
    if (np.diff(q, axis=1) < -1e-6 * np.maximum(1, np.abs(q[:, :-1]))).any():
        raise RuntimeError('CROSSED_QUANTILES')
    return q


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Yug:
    name = 'yug'
    device = 'cuda'

    def __init__(self, path, num_samples=100):
        from yug import YugPipeline
        self.pipeline = YugPipeline.from_pretrained(str(path), device_map='cuda')
        self.num_samples = num_samples
        cuda_sync()

    def torch_module(self):
        return self.pipeline.model

    def forecast(self, histories, horizon):
        import torch
        torch.manual_seed(SEED)
        context = [np.asarray(h, dtype=np.float32) for h in histories]
        f = self.pipeline.predict(context if len(context) > 1 else context[0], prediction_length=horizon,
                                  freq='D', seed=SEED, num_samples=self.num_samples, engine='cached')
        cuda_sync()
        if tuple(round(float(x), 6) for x in f.quantile_levels) != QUANTILES:
            raise RuntimeError('YUG_QUANTILE_LEVELS')
        values = np.asarray(f.values)  # (n, 9, h)
        return [v.T for v in values]


class TimesFM:
    name = 'timesfm-2.5'
    device = 'cuda'

    def __init__(self, path):
        import timesfm
        self.timesfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        self.model.load_checkpoint(str(path))
        self.key = None
        cuda_sync()

    def torch_module(self):
        return self.model.model

    def _compile(self, context, horizon, batch):
        key = (context, horizon, batch)
        if key != self.key:
            self.model.compile(self.timesfm.ForecastConfig(
                max_context=context, max_horizon=horizon, normalize_inputs=True,
                use_continuous_quantile_head=True, force_flip_invariance=True,
                infer_is_positive=True, fix_quantile_crossing=True, per_core_batch_size=batch))
            self.key = key

    def forecast(self, histories, horizon):
        import torch
        torch.manual_seed(SEED)
        longest = max(len(h) for h in histories)
        # Original protocol compiled max_context == context (256). The long arm
        # uses one fixed 2048 config; shorter inputs are left-padded and masked.
        context = 256 if longest <= 256 else MAX_CONTEXT
        self._compile(context, horizon, len(histories))
        with torch.inference_mode():
            _, q = self.model.forecast(horizon=horizon, inputs=[np.asarray(h, dtype=np.float32) for h in histories])
        cuda_sync()
        q = np.asarray(q)[:len(histories)]
        if q.shape[1:] != (horizon, 10):
            raise RuntimeError(f'TIMESFM_SHAPE {q.shape}')
        return [x[:, 1:] for x in q]


class ChronosBolt:
    name = 'chronos-bolt-base'
    device = 'cuda'

    def __init__(self, path):
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.pipeline = BaseChronosPipeline.from_pretrained(str(path), device_map='cuda', torch_dtype=torch.float32)
        if tuple(round(float(x), 6) for x in self.pipeline.quantiles) != QUANTILES:
            raise RuntimeError('CHRONOS_QUANTILE_LEVELS')
        cuda_sync()

    def torch_module(self):
        return self.pipeline.model

    def forecast(self, histories, horizon):
        torch = self.torch
        torch.manual_seed(SEED)
        ctx = [torch.tensor(np.asarray(h, dtype=np.float32)) for h in histories]
        with torch.inference_mode():
            q = self.pipeline.predict(ctx if len(ctx) > 1 else ctx[0][None, :], prediction_length=horizon)
        cuda_sync()
        return [x.T for x in q.cpu().numpy()]


def download_weights(key, root):
    """Fetch the pinned revision and verify safetensors bytes against the publisher LFS hash."""
    from huggingface_hub import HfApi, hf_hub_download
    pin = PINS[key]
    info = HfApi().model_info(pin['repo'], revision=pin['revision'], files_metadata=True)
    if info.sha != pin['revision']:
        raise RuntimeError('REVISION_MISMATCH')
    folder = Path(root) / key
    folder.mkdir(parents=True, exist_ok=True)
    receipts = []
    for sib in info.siblings:
        if sib.rfilename not in ('config.json', 'model.safetensors'):
            continue
        path = hf_hub_download(pin['repo'], sib.rfilename, revision=pin['revision'], local_dir=str(folder))
        digest = sha256(path)
        expected = sib.lfs.sha256 if sib.lfs else None
        if sib.rfilename.endswith('.safetensors') and digest != expected:
            raise RuntimeError('WEIGHT_HASH_MISMATCH')
        receipts.append(dict(file=sib.rfilename, bytes=os.path.getsize(path), sha256=digest,
                             publisher_sha256=expected, revision=pin['revision']))
    if {r['file'] for r in receipts} != {'config.json', 'model.safetensors'}:
        raise RuntimeError('WEIGHTS_INCOMPLETE')
    return folder, receipts


def pilot(model, rec):
    """Five synthetic weekly-seasonal series; must pass before any real window."""
    rng = np.random.default_rng(SEED)
    for n in range(5):
        h = 100 + 8 * np.sin(np.arange(256) * 2 * np.pi / 7) + rng.normal(0, 1, 256)
        q = check_forecast(model.forecast([h], 7)[0], 7)
        if not (80 < q[:, 4].mean() < 120):
            raise RuntimeError('PILOT_IMPLAUSIBLE')
    rec.log('pilot_passed', model=model.name)


def environment():
    import importlib.metadata as md
    env = dict(utc=utc(), python=platform.python_version(), platform=platform.platform(), machine=platform.machine())
    for p in ('torch', 'numpy', 'scipy', 'pandas', 'statsmodels', 'yug', 'timesfm', 'chronos-forecasting',
              'transformers', 'accelerate', 'huggingface-hub', 'safetensors'):
        try:
            env[p] = md.version(p)
        except Exception:
            env[p] = None
    for p in ('yug', 'timesfm', 'chronos-forecasting'):
        try:
            env[p + '_direct_url'] = json.loads(md.distribution(p).read_text('direct_url.json') or '{}')
        except Exception:
            env[p + '_direct_url'] = None
    try:
        import torch
        env['cuda'] = torch.version.cuda
        env['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env
