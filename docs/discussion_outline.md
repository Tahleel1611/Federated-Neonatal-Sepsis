# Discussion Outline (Research Paper)

## 1. Clinical Motivation

- Neonatal sepsis has subtle early physiological signatures and severe outcomes.
- Timely prediction can increase intervention window and reduce morbidity.

## 2. Data Silo Problem in Neonatal Care

- NICU data is fragmented across hospitals with heterogeneous protocols.
- Centralized pooling is constrained by policy, ethics, and governance.

## 3. FedNeo-Guard Contribution

- Transformer-LSTM hybrid captures both global and local physiological dynamics.
- Federated learning enables collaboration without centralizing raw patient data.
- Differential privacy lowers re-identification risk in transmitted updates.

## 4. Interpretability for Clinical Trust

- SHAP-based feature attributions identify key drivers (SpO2 drops, HRV changes).
- Attention maps provide temporal context for high-risk alerts.

## 5. Performance and Lead-Time Impact

- Report AUROC, sensitivity, specificity, and lead-time improvements.
- Compare centralized baseline vs federated + DP trade-offs.

## 6. Alarm Fatigue vs Privacy Trade-off

- Frame clinical deployment as an alert-budget optimization problem, not only AUROC optimization.
- Show that high sensitivity can be preserved under DP with class-weighted training, but specificity collapses as noise increases.
- Use the DP sweep to define a privacy-utility frontier with explicit operating points:
	- No-DP reference (Round 2): sensitivity 0.8067, specificity 0.5382.
	- DP noise 0.3 (best round): sensitivity 0.8044, specificity 0.4106, cumulative epsilon 76.5.
	- DP noise 0.6/0.9: sensitivity >= 0.80 but specificity < 0.20, clinically unsafe due to alarm burden.

## 7. Class-Weighted Noise Tolerance Contribution

- Emphasize that weighted BCE prevents loss of sensitivity under noisy DP-SGD updates.
- Present the proposed mechanism as a practical bridge between privacy guarantees and neonatal safety.
- Report privacy-loss ratio as a secondary criterion for policy decisions:
	- Delta Specificity / Delta Epsilon relative to no-DP operating point.
	- Identify inflection where additional privacy causes disproportionate specificity loss.

## 8. Limitations

- Synthetic-first development does not replace real-world external validation.
- Current DP settings reveal a steep utility penalty in specificity at tighter privacy levels.
- SMPC integration is placeholder and requires production cryptographic implementation.

## 9. Future Work

- Real MIMIC extraction and prospective multi-center validation.
- Full secure aggregation/SMPC stack.
- Adaptive threshold scheduling based on NICU staffing and alert budget constraints.
- Edge optimization for Jetson Nano deployment (quantization and pruning).

## 10. Recommended Deployment Policy

- Define deployment by operational tier, not a single static privacy parameter.

### Tier 1: High-Surge Mode (Maximum Catch)

- Use no-DP or very low-noise configuration during periods with full staffing and high sepsis concern.
- Clinical objective: maximize early catch rate; tolerate higher false alerts.
- Governance: short-duration emergency use with explicit audit logging.

### Tier 2: Standard Care (Clinical Champion)

- Use DP noise multiplier 0.3 as default bedside setting.
- Operating profile: tuned sensitivity >= 0.80 with specificity around 0.41 in current sweep.
- Governance: balance patient-safety utility and privacy posture for routine care.

### Tier 3: Privacy-Critical Research Mode

- Use higher-noise DP settings (for example 0.6 to 0.9) only for retrospective or offline research workflows.
- Clinical limitation: specificity collapse increases alert fatigue risk, so not recommended for live bedside alerts.
- Governance: prioritize de-identification strength where legal/privacy requirements dominate over real-time precision.

### Hospital Admin Playbook

- Review monthly alert burden (false alerts per bed-day), sensitivity, and staffing capacity.
- Select tier using explicit thresholds: move up for surge detection, move down for alarm fatigue control.
- Require sign-off from NICU lead + privacy/compliance team when switching tiers.