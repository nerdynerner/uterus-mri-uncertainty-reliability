"""Preprocessing and training implementation used for the six completed models."""
import argparse
import functools
import importlib.metadata
import itertools
import json
import math
import platform
import random
import resource
import time
from pathlib import Path
from study_manifest import sha256, validate_manifest, validate_splits, within, write_new


def dependencies():
    global np, nib, torch, resample_from_to, model
    import numpy as np
    import nibabel as nib
    import torch
    from nibabel.processing import resample_from_to
    from monai_components import model


def checked_image(path):
    image = nib.load(path)
    if len(image.shape) != 3 or min(image.shape) < 1:
        raise ValueError('Only audited 3D volumes are supported')
    affine = image.affine
    norms = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(affine).all() or np.any(norms <= 0):
        raise ValueError('Invalid affine')
    direction = affine[:3, :3] / norms
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError('Sheared grids require a separately verified distance implementation')
    # Conflicting coded transforms require explicit adjudication, not silent choice.
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if qc and sc and not np.allclose(q, s, atol=1e-3):
        raise ValueError('Conflicting qform and sform')
    return image


def prepare_image(path):
    native = checked_image(path)
    canonical = nib.as_closest_canonical(native)
    spacing = np.linalg.norm(canonical.affine[:3, :3], axis=0)
    # Keep acquired through-plane spacing and direction. RAS canonical voxel
    # axis 2 must correspond to the audited axial stack, including obliquity.
    direction = canonical.affine[:3, :3] / spacing
    if int(np.argmax(np.abs(direction[:, 2]))) != 2:
        raise ValueError('Axial-stack requirement failed')
    target_spacing = np.array([1.5, 1.5, spacing[2]])
    target_shape = tuple((np.ceil((np.array(canonical.shape)-1)*spacing/target_spacing)+1).astype(int))
    target_affine = canonical.affine.copy()
    target_affine[:3, :3] = direction * target_spacing
    resampled = resample_from_to(canonical, (target_shape, target_affine), order=1)
    x = np.array(resampled.dataobj, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite image intensities')
    foreground = x != 0
    if foreground.sum() < 2:
        raise ValueError('Image has insufficient nonzero intensities')
    low, high = np.percentile(x[foreground], [1, 99])
    # NumPy percentile bounds can promote clipping to float64. Keep the model
    # input explicitly float32, to avoid version-specific promotion.
    x = np.clip(x, low, high).astype(np.float32, copy=False)
    std = float(x[foreground].std())
    if std <= 1e-8:
        raise ValueError('Constant image cannot be normalized')
    x[foreground] = (x[foreground]-x[foreground].mean())/std
    x[~foreground] = 0
    # No ROI crop or positive-slice selection. Padding is removed before inversion.
    h, w = x.shape[:2]
    pad = ((0, (-h) % 8), (0, (-w) % 8), (0, 0))
    x = np.pad(x, pad, mode='constant')
    return x, native, (target_shape, target_affine), (h, w)


def prepare_mask(path, native, target):
    reference = checked_image(path)
    if reference.shape != native.shape or not np.allclose(reference.affine, native.affine, atol=1e-3):
        raise ValueError('Manifest-approved pair no longer shares the verified native grid')
    data = np.asarray(reference.dataobj)
    if not np.isfinite(data).all() or not set(np.unique(data)).issubset({0, 1}):
        raise ValueError('Binary uterus masks only; multiclass conversion requires documented mapping')
    result = np.asarray(resample_from_to(reference, target, order=0).dataobj, dtype=np.float32)
    h, w = result.shape[:2]
    return np.pad(result, ((0, (-h) % 8), (0, (-w) % 8), (0, 0)))


def native_probability(probability, native, target, original_hw):
    h, w = original_hw
    prob_image = nib.Nifti1Image(probability[:h, :w].astype(np.float32), target[1])
    restored = resample_from_to(prob_image, (native.shape, native.affine), order=1)
    return np.clip(np.asarray(restored.dataobj, dtype=np.float32), 0, 1)


def dice(a, b):
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2*int(np.count_nonzero(a & b))/total


def native_metrics(prediction, reference_path, native=None):
    reference = checked_image(reference_path)
    if native is not None and (native.shape != reference.shape or not np.allclose(native.affine, reference.affine, atol=1e-3)):
        raise ValueError('Reference does not occupy the native prediction grid')
    y = np.asarray(reference.dataobj) > 0
    if prediction.shape != y.shape:
        raise ValueError('Native prediction/reference shapes differ')
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000
    union = int(np.count_nonzero(prediction | y))
    record = {'dice': dice(prediction, y),
              'iou': 1.0 if union == 0 else int(np.count_nonzero(prediction & y))/union,
              'absolute_volume_error_ml': abs(int(prediction.sum())-int(y.sum()))*voxel_ml,
              'prediction_empty': not bool(prediction.any()), 'reference_empty': not bool(y.any())}
    # Surface endpoints are deliberately not approximated with voxelwise distances.
    # Their MONAI implementation must be integration-tested before publication.
    return record


def device_and_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def predict_volume(net, x, device, stochastic=False):
    net.eval()
    if stochastic:
        layers = [m for m in net.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)) and m.p > 0]
        if not layers:
            raise ValueError('Dropout checkpoint has no active dropout modules')
        for layer in layers:
            layer.train()
    probability = np.empty(x.shape, dtype=np.float32)
    try:
        with torch.inference_mode():
            for z in range(x.shape[2]):
                tensor = torch.from_numpy(np.array(x[:, :, z], copy=True))[None, None].to(device)
                probability[:, :, z] = torch.sigmoid(net(tensor))[0, 0].cpu().numpy()
    finally:
        net.eval()
    return probability


def primary_reference(row):
    return next(m['path'] for m in row['masks'] if m['rater'] == row['training_rater'])


def train(args, cases, partitions):
    benchmark = json.loads(args.benchmark.read_text())
    integration = json.loads(args.integration.read_text())
    if benchmark.get('status') != 'passed' or integration.get('status') != 'passed':
        raise ValueError('Passing resource and synthetic integration records required')
    if integration.get('pipeline_sha256') != sha256(Path(__file__)):
        raise ValueError('Rerun synthetic integration after any pipeline change')
    if benchmark.get('peak_process_rss_bytes', math.inf) > 3*1024**3:
        raise ValueError('Benchmark process RSS exceeds the conservative 3 GiB starting limit')
    if args.seed not in (61901, 61902, 61903, 61904, 61905, 61911):
        raise ValueError('Use a prespecified independent model seed')
    if (args.seed == 61911) != (args.dropout == .2):
        raise ValueError('Seed 61911 is reserved for dropout=0.2')
    args.output.mkdir(parents=True, exist_ok=False)
    device = device_and_seed(args.seed)
    net = model(args.dropout).to(device)
    from monai.losses import DiceCELoss
    objective = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    indexed = {r['participant']: r for r in cases}
    training = [indexed[p] for p in partitions['train']]
    validation = [indexed[p] for p in partitions['validation']]
    # Cache at most one volume; SSD .npy avoids retaining the whole cohort in RAM.
    cache = args.output/'prepared_train'
    cache.mkdir()
    samples = []
    for i, row in enumerate(training):
        x, native, target, _ = prepare_image(within(args.root, row['image']))
        if x.shape[0]*x.shape[1] > 512*512:
            raise ValueError('Slice exceeds initial resource envelope; benchmark this actual shape before changing the plan')
        y = prepare_mask(within(args.root, primary_reference(row)), native, target)
        np.save(cache/f'{i}_x.npy', x)
        np.save(cache/f'{i}_y.npy', y)
        samples.append(x.shape[2])
    del x, y
    @functools.lru_cache(maxsize=1)
    def volume(index):
        return np.load(cache/f'{index}_x.npy', mmap_mode='r'), np.load(cache/f'{index}_y.npy', mmap_mode='r')
    class Slices(torch.utils.data.Dataset):
        def __len__(self):
            return sum(samples)
        def __getitem__(self, unused):
            # Uniform participant then uniform acquired slice, including negatives.
            i = random.randrange(len(training))
            x, y = volume(i)
            z = random.randrange(x.shape[2])
            return (torch.from_numpy(np.array(x[:, :, z], copy=True))[None],
                    torch.from_numpy(np.array(y[:, :, z], copy=True))[None])
    def collate(batch):
        height = max(pair[0].shape[1] for pair in batch)
        width = max(pair[0].shape[2] for pair in batch)
        result = []
        for index in (0, 1):
            result.append(torch.stack([torch.nn.functional.pad(pair[index],
                (0, width-pair[index].shape[2], 0, height-pair[index].shape[1])) for pair in batch]))
        # A validity map prevents variable-size batch padding from affecting loss.
        valid = torch.stack([torch.nn.functional.pad(torch.ones_like(pair[1]),
                (0, width-pair[1].shape[2], 0, height-pair[1].shape[1])) for pair in batch])
        return result[0], result[1], valid
    loader = torch.utils.data.DataLoader(Slices(), batch_size=args.batch_size,
                                        num_workers=0, collate_fn=collate)
    history, best, wait = [], -1.0, 0
    config = {'seed': args.seed, 'dropout': args.dropout, 'batch_size': args.batch_size,
              'epochs': args.epochs, 'patience': 15, 'augmentation': 'none in initial implementation',
              'manifest_sha256': sha256(args.manifest), 'split_sha256': sha256(args.splits),
              'code_sha256': sha256(Path(__file__)), 'components_sha256': sha256(Path(__file__).with_name('monai_components.py')),
              'benchmark_sha256': sha256(args.benchmark), 'integration_sha256': sha256(args.integration),
              'device': str(device), 'platform': platform.platform(),
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'monai', 'numpy', 'nibabel', 'scipy')}}
    write_new(args.output/'configuration.json', config)
    for epoch in range(args.epochs):
        start = time.perf_counter()
        net.train()
        losses = []
        for x, y, valid in loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            # Apply DiceCE to each true image rectangle, so padding is not scored.
            terms = []
            for i in range(len(x)):
                h = int(valid[i, 0].any(dim=1).sum())
                w = int(valid[i, 0].any(dim=0).sum())
                terms.append(objective(logits[i:i+1, :, :h, :w], y[i:i+1, :, :h, :w]))
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise RuntimeError('Nonfinite gradients')
            optimizer.step()
            losses.append(float(loss.detach()))
        values = []
        for row in validation:
            x, native, target, hw = prepare_image(within(args.root, row['image']))
            native_prob = native_probability(predict_volume(net, x, device), native, target, hw)
            values.append(native_metrics(native_prob >= .5, within(args.root, primary_reference(row)), native)['dice'])
        score = float(np.mean(values))
        sync(device)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if platform.system() == 'Darwin' else rss*1024
        history.append({'epoch': epoch+1, 'train_loss': float(np.mean(losses)),
                        'validation_mean_native_volume_dice': score,
                        'seconds': time.perf_counter()-start, 'peak_process_rss_bytes': rss})
        write_new(args.output/f'history_epoch_{epoch+1:03}.json', history[-1])
        if score > best:
            best, wait = score, 0
            # New checkpoint names preserve every earlier selected model.
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'configuration': config,
                        'epoch': epoch+1, 'validation_dice': score}, args.output/f'best_epoch_{epoch+1:03}.pt')
        else:
            wait += 1
        if rss > 4*1024**3:
            raise RuntimeError('Measured process RSS exceeds 4 GiB; stop and revise resource plan')
        if wait >= 15:
            break
    selected = max(history, key=lambda r: r['validation_mean_native_volume_dice'])
    write_new(args.output/'selected_checkpoint.json', {
        'path': f"best_epoch_{selected['epoch']:03}.pt", 'epoch': selected['epoch'],
        'sha256': sha256(args.output/f"best_epoch_{selected['epoch']:03}.pt")})


def infer(args, cases, partitions):
    import statistics
    if args.partition in ('internal_test', 'external_test'):
        if not args.frozen_policy:
            raise ValueError('Held-out inference requires the already frozen calibration policy')
        policy = json.loads(args.frozen_policy.read_text())
        if policy.get('provenance', {}).get('checkpoint_sha256', {}).get(args.method) != [sha256(p) for p in args.checkpoints]:
            raise ValueError('Frozen policy must bind these exact model checkpoints')
        if set(policy.get('calibration_ids', [])) != set(partitions['calibration']):
            raise ValueError('Calibration ID provenance mismatch')
    checkpoints = [torch.load(p, map_location='cpu', weights_only=True) for p in args.checkpoints]
    seeds = [c['configuration']['seed'] for c in checkpoints]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Independent model seeds required')
    if args.method == 'ensemble' and set(seeds) != set(range(61901, 61906)):
        raise ValueError('Primary ensemble requires all five prespecified models')
    if args.method == 'dropout' and seeds != [61911]:
        raise ValueError('Dropout inference requires the separately trained dropout model')
    if args.method == 'deterministic' and seeds != [61901]:
        raise ValueError('Deterministic comparator uses the first ensemble member')
    for checkpoint in checkpoints:
        c = checkpoint['configuration']
        if c['manifest_sha256'] != sha256(args.manifest) or c['split_sha256'] != sha256(args.splits):
            raise ValueError('Checkpoint data provenance mismatch')
        expected = .2 if args.method == 'dropout' else 0.
        if c['dropout'] != expected:
            raise ValueError('Checkpoint architecture mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    device = device_and_seed(61930)
    rows = []
    for row in cases:
        if row['participant'] not in partitions[args.partition]:
            continue
        # Opaque ordinal folder names avoid using participant text as a path.
        folder = args.output/f'case_{len(rows):04}'
        folder.mkdir()
        x, native, target, hw = prepare_image(within(args.root, row['image']))
        repeats = 30 if args.method == 'dropout' else len(checkpoints)
        members = np.lib.format.open_memmap(folder/'member_masks.npy', mode='w+', dtype=np.bool_,
                                            shape=(repeats, *native.shape))
        mean = np.zeros(native.shape, dtype=np.float32)
        for index in range(repeats):
            checkpoint = checkpoints[0] if args.method == 'dropout' else checkpoints[index]
            net = model(checkpoint['configuration']['dropout'])
            net.load_state_dict(checkpoint['state_dict'])
            net.to(device)
            if args.method == 'dropout':
                # Different reproducible stochastic stream per pass.
                torch.manual_seed(61930+index)
            probability = predict_volume(net, x, device, stochastic=args.method == 'dropout')
            p = native_probability(probability, native, target, hw)
            mean += p/repeats
            members[index] = p >= .5
            del net, probability, p
            if device.type == 'mps':
                torch.mps.empty_cache()
        prediction = mean >= .5
        empty = not bool(prediction.any())
        raw_score = None
        both_empty_pairs = 0
        if repeats > 1:
            scores = []
            for a, b in itertools.combinations(range(repeats), 2):
                aa, bb = members[a], members[b]
                both_empty_pairs += int(not aa.any() and not bb.any())
                scores.append(1-dice(aa, bb))
            raw_score = statistics.fmean(scores)
            uncertainty = raw_score
        else:
            safe = np.clip(mean, 1e-7, 1-1e-7)
            entropy = -(safe*np.log2(safe)+(1-safe)*np.log2(1-safe))
            raw_score = float(entropy[prediction].mean()) if not empty else None
            uncertainty = 1.0 if empty else raw_score
        references = []
        # Calibration never opens reference labels; scores alone set the threshold.
        if args.partition != 'calibration':
            for mask in row['masks']:
                references.append({'rater': mask['rater'], **native_metrics(prediction, within(args.root, mask['path']), native)})
        record = {'participant': row['participant'], 'method': args.method,
                  'partition': args.partition, 'uncertainty': uncertainty,
                  'raw_uncertainty': raw_score, 'empty_prediction': empty,
                  'both_empty_member_pairs': both_empty_pairs,
                  'references': references, 'errors': [1-r['dice'] for r in references]}
        np.save(folder/'mean_native_probability.npy', mean)
        np.save(folder/'native_affine.npy', native.affine)
        write_new(folder/'metrics.json', record)
        members.flush()
        del members
        rows.append(record)
    write_new(args.output/'participant_outputs.json', rows)
    write_new(args.output/'provenance.json', {'checkpoint_sha256': [sha256(p) for p in args.checkpoints],
        'manifest_sha256': sha256(args.manifest), 'splits_sha256': sha256(args.splits),
        'code_sha256': sha256(Path(__file__)), 'method': args.method,
        'scope': 'Native overlap and volume endpoints; extended endpoints are evaluated separately'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'infer'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--integration', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=61901)
    parser.add_argument('--dropout', type=float, choices=[0., .2], default=0.)
    parser.add_argument('--partition', choices=['calibration', 'internal_test', 'external_test'])
    parser.add_argument('--method', choices=['deterministic', 'dropout', 'ensemble'])
    parser.add_argument('--checkpoints', nargs='+', type=Path)
    parser.add_argument('--frozen-policy', type=Path)
    args = parser.parse_args()
    if args.command == 'train' and (not args.benchmark or not args.integration):
        parser.error('Training requires --benchmark and --integration records')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Positive epochs and batch size required')
    if args.command == 'infer' and (not args.method or not args.partition or not args.checkpoints):
        parser.error('Inference requires --method, --partition, --checkpoints')
    cases = validate_manifest(json.loads(args.manifest.read_text()), args.root)
    split = json.loads(args.splits.read_text())
    if split.get('manifest_sha256') != sha256(args.manifest):
        raise ValueError('Split was generated for a different manifest')
    partitions = validate_splits(cases, split)
    dependencies()
    if args.command == 'train':
        train(args, cases, partitions)
    else:
        infer(args, cases, partitions)


if __name__ == '__main__':
    main()
