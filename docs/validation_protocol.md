# Clinical Validation Protocol

## Objective

Evaluate neonatal sepsis prediction within a 6-hour horizon for privacy-preserving federated training.

## Primary Metrics

- AUROC
- Sensitivity (Recall for sepsis class)
- Specificity
- Lead Time (hours between first high-risk alert and clinical diagnosis)

## Evaluation Plan

1. Compute patient-level predictions every step using the model probability.
2. Mark high-risk alerts using a threshold (default 0.5, then tune by Youden's index).
3. For each patient with sepsis diagnosis:
   - Find first timestamp with high-risk alert.
   - Compute lead time = diagnosis_time - first_alert_time.
4. Report:
   - Mean/median lead time
   - Bootstrapped 95% CI for AUROC, sensitivity, specificity
   - Performance by gestational-age strata

## Publication-Ready Checks

- Calibration curve and Brier score
- External holdout (hospital-wise) robustness
- Error review on false negatives with clinician input