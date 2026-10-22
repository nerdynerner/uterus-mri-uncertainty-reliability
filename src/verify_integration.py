"""Synthetic geometry and gradient checks, never patient performance evidence."""
import argparse
import json
import tempfile
from pathlib import Path
from study_manifest import sha256, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    report = {'clinical_data_used': False, 'scope': 'synthetic geometry and gradient checks'}
    try:
        import pipeline as p
        p.dependencies()
        np, nib, torch = p.np, p.nib, p.torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=args.scratch) as temporary:
            folder = Path(temporary)
            # A flipped native grid with anisotropic slices tests both orientation
            # recovery and physical-space resampling without assuming array order.
            affine = np.array([[-1.5, 0, 0, 57], [0, 1.5, 0, -18], [0, 0, 4, 11], [0, 0, 0, 1.]])
            shape = (40, 48, 7)
            volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)+1
            label = np.zeros(shape, dtype=np.uint8)
            label[11:29, 16:33, 2:5] = 1
            image_path, label_path = folder/'image.nii.gz', folder/'mask.nii.gz'
            nib.save(nib.Nifti1Image(volume, affine), image_path)
            nib.save(nib.Nifti1Image(label, affine), label_path)
            x, native, target, hw = p.prepare_image(image_path)
            transformed = p.prepare_mask(label_path, native, target)
            assert x.dtype == np.float32, 'Preprocessed model input must be float32'
            assert transformed.dtype == np.float32, 'Preprocessed target must be float32'
            restored = p.native_probability(transformed, native, target, hw)
            assert np.array_equal(restored >= .5, label.astype(bool)), 'Native orientation round-trip failed'
            assert x.shape[:2] == transformed.shape[:2]
            # Wrong sequence/grid must fail without silent alignment.
            shifted = affine.copy()
            shifted[0, 3] += 3
            wrong = folder/'wrong.nii.gz'
            nib.save(nib.Nifti1Image(label, shifted), wrong)
            try:
                p.prepare_mask(wrong, native, target)
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched mask grid was accepted')
            from monai.losses import DiceCELoss
            net = p.model(.2)
            actual_input = torch.from_numpy(np.ascontiguousarray(x[:,:,3]))[None,None]
            actual_target = torch.from_numpy(np.ascontiguousarray(transformed[:,:,3]))[None,None]
            actual_loss = DiceCELoss(sigmoid=True)(net(actual_input),actual_target)
            assert torch.isfinite(actual_loss), 'Preprocessed image/target forward pass failed'
            tensor = torch.randn(1, 1, 64, 64)
            loss = DiceCELoss(sigmoid=True)(net(tensor), torch.zeros_like(tensor))
            loss.backward()
            assert torch.isfinite(loss)
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in net.parameters())
            from monai_components import stochastic_probability
            probabilities = stochastic_probability(net, tensor, passes=3)
            assert probabilities.shape == (3, 1, 1, 64, 64)
            assert not torch.equal(probabilities[0], probabilities[1]), 'Dropout passes identical'
            report.update(status='passed', native_roundtrip_dice=p.dice(restored >= .5, label > 0),
                          gradient_finite=True, dropout_stochastic=True, mismatch_rejected=True,
                          preprocessed_input_float32=True, preprocessed_input_forward_pass=True,
                          pipeline_sha256=sha256(Path(p.__file__)))
    except Exception as error:
        report.update(status='blocked_or_failed', error=repr(error))
    write_new(args.output, report)
    print(json.dumps(report, indent=2))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
