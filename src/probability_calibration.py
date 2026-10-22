"""Secondary probability calibration; never changes the review-policy scores.

Fit only source calibration participants. Sampling uses image-derived nonzero
voxels, never label-derived positive crops. Weight participants equally.
Temperature of an averaged probability is an operational recalibration, not
calibration of individual ensemble members or proof of a safe review policy.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def sample_patient(probabilities, label, image, seed, maximum=4096):
    if probabilities.shape != label.shape or label.shape != image.shape:
        raise ValueError('Native prediction/reference/image shapes must match')
    if not np.isfinite(probabilities).all() or not np.isfinite(image).all():
        raise ValueError('Finite inputs required')
    if not np.isin(label,[0,1]).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('Binary target and probabilities in [0,1] required')
    available = np.flatnonzero(image.ravel() != 0)
    if not len(available):
        raise ValueError('Empty image-derived sampling region')
    chosen = np.random.default_rng(seed).choice(available,size=min(maximum,len(available)),replace=False)
    return {'probabilities': probabilities.ravel()[chosen].astype(np.float64),
            'labels': label.ravel()[chosen].astype(np.float64),
            'sampled_voxels': len(chosen), 'eligible_image_voxels': len(available)}


def calibrated(probability, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    p = np.clip(probability,1e-7,1-1e-7)
    logits = (np.log(p)-np.log1p(-p))/temperature
    return 1/(1+np.exp(-np.clip(logits,-700,700)))


def patient_nll(row, temperature):
    p = np.clip(calibrated(row['probabilities'],temperature),1e-12,1-1e-12)
    y = row['labels']
    return float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())


def fit_temperature(calibration_patients):
    if not calibration_patients:
        raise ValueError('No calibration patients')
    objective = lambda log_t: float(np.mean([patient_nll(r,np.exp(log_t)) for r in calibration_patients]))
    result = minimize_scalar(objective,bounds=(np.log(.05),np.log(20.)),method='bounded')
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Temperature optimization failed')
    t = float(np.exp(result.x))
    # Include T=1 explicitly so flat or numerically worse fits cannot worsen NLL.
    if objective(0.) <= result.fun:
        t = 1.
    return {'temperature':t, 'n_calibration_participants':len(calibration_patients),
            'temperature_bounds':[.05,20.],
            'before_participant_balanced_nll':objective(0.),
            'after_participant_balanced_nll':float(np.mean([patient_nll(r,t) for r in calibration_patients])),
            'scope':'Source calibration fit only; preserves 0.5 segmentation threshold; no conformal guarantee'}


def evaluate(patient_rows, temperature=1.):
    if not patient_rows:
        raise ValueError('No evaluation participants')
    bins = [{'weight':0.,'probability_sum':0.,'positive_sum':0.} for _ in range(10)]
    briers, nlls = [], []
    for row in patient_rows:
        p,y = calibrated(row['probabilities'],temperature),row['labels']
        briers.append(float(((p-y)**2).mean()))
        nlls.append(patient_nll(row,temperature))
        assignment = np.minimum((p*10).astype(int),9)
        voxel_weight = 1/(len(patient_rows)*len(p))
        for i in range(10):
            mask = assignment == i
            bins[i]['weight'] += float(mask.sum()*voxel_weight)
            bins[i]['probability_sum'] += float(p[mask].sum()*voxel_weight)
            bins[i]['positive_sum'] += float(y[mask].sum()*voxel_weight)
    output_bins = []
    for i,b in enumerate(bins):
        weight = b['weight']
        output_bins.append({'lower':i/10,'upper':(i+1)/10,'weight':weight,
                            'mean_predicted_probability':b['probability_sum']/weight if weight else None,
                            'observed_target_fraction':b['positive_sum']/weight if weight else None})
    ece = sum(b['weight']*abs(b['mean_predicted_probability']-b['observed_target_fraction'])
              for b in output_bins if b['weight'])
    return {'n_participants':len(patient_rows),'temperature':temperature,
            'participant_balanced_brier':float(np.mean(briers)),
            'participant_balanced_nll':float(np.mean(nlls)),
            'participant_balanced_ece_10_bins':ece,'reliability_bins':output_bins,
            'scope':'Target probabilities within sampled image-derived region; not failure-detection quality'}


if __name__ == '__main__':
    import argparse,json
    from study_manifest import write_new
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    label = np.array([0,1,0,1],dtype=np.uint8).reshape(2,2,1)
    probability = np.array([.1,.9,.2,.8]).reshape(2,2,1)
    row = sample_patient(probability,label,np.ones_like(probability),seed=61941)
    fitted = fit_temperature([row])
    assert fitted['after_participant_balanced_nll'] <= fitted['before_participant_balanced_nll']
    result = evaluate([row],fitted['temperature'])
    assert abs(sum(b['weight'] for b in result['reliability_bins'])-1.) < 1e-12
    assert np.array_equal(calibrated(probability,fitted['temperature'])>=.5,probability>=.5)
    record = {'status':'passed','clinical_data_used':False,'calibration_fit':fitted,
              'sample_reproducible':np.array_equal(row['probabilities'],sample_patient(probability,label,np.ones_like(probability),61941)['probabilities']),
              'scope':'Artificial fixture only; not an estimated patient calibration effect'}
    write_new(args.output,record)
    print(json.dumps(record,indent=2))
