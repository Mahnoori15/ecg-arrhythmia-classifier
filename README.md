# ECG Arrhythmia Classifier

![CI](https://github.com/Mahnoori15/ecg-arrhythmia-classifier/actions/workflows/ci.yml/badge.svg)

Classifies individual heartbeats from ECG recordings into the five **AAMI EC57** classes using a 1D convolutional neural network plus RR-interval rhythm features. The model is tested **inter-patient**: it is trained on one group of patients and evaluated on completely different patients.

> Research and education project. Not a medical device.

## Why inter-patient evaluation matters
Many published ECG models report ~99% accuracy, but they often split beats randomly. The same patient's heartbeats then appear in both training and test sets, so the model memorises patients instead of learning arrhythmias. Real deployments see new patients, so this project uses the standard **de Chazal DS1/DS2 split** (22 training records, 22 test records, no overlap). This is harder, and the scores are honest.

## Classes
| AAMI class | Meaning | MIT-BIH labels |
|---|---|---|
| N | Normal and bundle branch block beats | N, L, R, e, j |
| S | Supraventricular ectopic beats | A, a, J, S |
| V | Ventricular ectopic beats | V, E |
| F | Fusion beats | F |
| Q | Paced / unclassifiable | /, f, Q |

## Pipeline
1. **Download** the MIT-BIH Arrhythmia Database from PhysioNet (`wfdb`)
2. **Filter** lead MLII with a 0.5–40 Hz Butterworth band-pass to remove baseline wander and noise
3. **Segment** a 256-sample window (~0.71 s) around each annotated R peak, z-scored per beat
4. **RR features**: pre-RR, post-RR, local mean RR and pre/post ratio, normalised by each patient's mean RR so they transfer across patients. This matters most for S beats, which look normal but arrive early.
5. **Model**: 4-block 1D CNN for beat shape, concatenated with RR features into a small classifier head
6. **Training**: class-weighted cross-entropy (the classes are heavily imbalanced), early stopping on macro-F1 using held-out DS1 patients
7. **Evaluation** on unseen DS2 patients: per-class precision, sensitivity, F1 and a confusion matrix

## Results (DS2, unseen patients)
Run `python -m ecg.train` to reproduce. Metrics are saved to `results/metrics.json` and the plot to `results/confusion_matrix.png`.

| Class | Precision | Sensitivity | F1 |
|---|---|---|---|
| N | – | – | – |
| S | – | – | – |
| V | – | – | – |
| F | – | – | – |
| **Macro F1** | | | **–** |

<!-- After training, commit results/confusion_matrix.png and uncomment:
![Confusion matrix](results/confusion_matrix.png) -->

## Quick start
```bash
pip install -e ".[dev]"
python -m ecg.train --epochs 15   # downloads ~100 MB of data the first time; runs on CPU
pytest -q                         # tests use synthetic ECG, no download needed
```

## Limitations and next steps
- Single lead (MLII) and a single 1990s dataset. Next: test generalisation on the INCART or PTB-XL datasets.
- F and Q classes have very few test beats, so their scores are unstable.
- Add explainability (e.g. saliency maps over the beat) so clinicians can see which part of the waveform drove a prediction.

## References
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol* 2001.
- de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE TBME* 2004.
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 2000.
