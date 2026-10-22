# Methodological and computational notes

The analytical cohort, partitions, checkpoint selections and review thresholds are fixed records of the completed study. Source identifiers describe anonymised archive entries and are not independently verified clinical identities.

## Implementation lineage

Training source and MONAI model-definition hashes were identical across the six completed model runs. The selected checkpoint files were checked against their recorded SHA-256 hashes against the computational records. The model epochs were 35/34/30/40/33/60, with selected epochs 20/19/15/25/18/45.

The evaluated inference implementation uses packed-mask pairwise disagreement. The retained training implementation uses direct mask comparisons in its inference function. Training calculations are identical between the two modules; the publication workflow uses the evaluated inference implementation.

Two secondary implementation amendments occurred after held-out inference began. The reference loader affine tolerance was aligned to the admission and primary evaluation tolerance of 0.001. Reference-disagreement calculations removed background outside the unchanged reference-union-plus-5-mm domain. Recorded full-grid comparisons covered three artificial fixtures and the external case with three references, with tolerance 1e-12. These amendments did not alter cohort admission, predictions, primary review thresholds or statistical endpoints. Supporting numerical records are retained in `results/`.

The portable implementation resolves its workspace through `UTERUS_MRI_WORKSPACE`. `config/source_lineage.json` maps computational source hashes to the distributed modules. Source image paths, image and mask hashes, eligibility flags and participant assignments match the completed analysis. The fixed partition record binds `config/manifest.json`; measured-run records retain the corresponding execution hashes.

## Interpretation and reproduction limits

External and internal cohorts differ in acquisition, image contrast and reference practices. The study does not isolate a causal scanner effect. The calibration sample contains eight cases. Bootstrap intervals condition on the trained models; undefined draws are reported when no case is retained. Secondary endpoints are descriptive.

The preserved numerical verification record covers 111 held-out participant-method native outputs. Repository checks additionally recompute the score-level primary analysis. Native surface, probability and disagreement endpoints require source images and predictions; their recorded aggregate results do not constitute a new image-level reproduction. Fresh-environment installation, full retraining and independent cross-hardware replication have not been established.

