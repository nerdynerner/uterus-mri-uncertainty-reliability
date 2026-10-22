"""Artificial resampling and checkpoint checks, without clinical inference."""
import argparse
import json
import math
import tempfile
from pathlib import Path
from study_manifest import sha256, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    record = {'clinical_data_used': False, 'scope': 'artificial changed-spacing/oblique geometry and save/load'}
    try:
        import pipeline as p
        p.dependencies()
        np, nib, torch = p.np, p.nib, p.torch
        torch.set_num_threads(2)
        checks = []
        with tempfile.TemporaryDirectory(dir=args.scratch) as directory:
            folder = Path(directory)
            for angle in [0., 12.]:
                radians = math.radians(angle)
                rotation = np.array([[math.cos(radians), 0, math.sin(radians)],
                                     [0, 1, 0], [-math.sin(radians), 0, math.cos(radians)]])
                affine = np.eye(4)
                affine[:3, :3] = rotation @ np.diag([1., 1.2, 4.])
                affine[:3, 3] = [17., -23., 9.]
                shape = (49, 53, 9)
                i, j, k = np.indices(shape)
                mask = (((i-24)/15)**2+((j-26)/16)**2+((k-4)/3)**2 <= 1).astype(np.uint8)
                image = (i+j*2+k*3+1).astype(np.float32)
                image_path = folder/f'image_{angle}.nii.gz'
                mask_path = folder/f'mask_{angle}.nii.gz'
                nib.save(nib.Nifti1Image(image, affine), image_path)
                nib.save(nib.Nifti1Image(mask, affine), mask_path)
                x, native, target, hw = p.prepare_image(image_path)
                transformed = p.prepare_mask(mask_path, native, target)
                restored = p.native_probability(transformed, native, target, hw)
                value = p.dice(restored >= .5, mask > 0)
                assert value > .90, 'Excessive synthetic resampling distortion'
                assert restored.shape == shape
                assert np.allclose(target[1][:3, 2], affine[:3, 2]), 'Through-plane direction changed'
                assert np.allclose(np.linalg.norm(target[1][:3, :3], axis=0), [1.5, 1.5, 4.])
                checks.append({'obliquity_degrees': angle, 'restored_dice': value,
                               'native_shape': list(shape), 'processed_shape': list(x.shape)})
            torch.manual_seed(61900)
            network = p.model().eval()
            sample = torch.randn(1, 1, 64, 72)
            with torch.inference_mode():
                before = network(sample)
                torch.save({'state_dict': network.state_dict()}, folder/'synthetic.pt')
                loaded = p.model().eval()
                loaded.load_state_dict(torch.load(folder/'synthetic.pt', weights_only=True)['state_dict'])
                assert torch.equal(before, loaded(sample)), 'CPU checkpoint prediction changed'
            record.update(status='passed', geometry=checks, checkpoint_exact_cpu=True,
                          pipeline_sha256=sha256(Path(p.__file__)),
                          limitation='Simple synthetic objects; not evidence of anatomical pairing or clinical accuracy')
    except Exception as error:
        record.update(status='blocked_or_failed', error=repr(error))
    write_new(args.output, record)
    print(json.dumps(record, indent=2))
    if record['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
