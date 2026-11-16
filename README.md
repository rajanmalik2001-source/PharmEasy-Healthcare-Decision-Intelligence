Domain: Healthcare

# PharmEasy Healthcare Decision Intelligence

## How to run

From a clean clone, run these commands in order from the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The scripts use Python 3.10+ and NumPy 2.2.6. The generator uses only the Python standard library. Each Part writes its committed result files under `outputs/`.

## Part 1 findings

The estimated mean care-plan cost is INR 8.9382 thousand (about INR 8,938), with a 95% confidence interval of INR 8.3026–9.5738 thousand. The test rejects H0: μ = 8.0 because z = 2.8930 exceeds the two-tailed critical value 1.96. A Type I error here would mean concluding that the average care-plan cost differs from INR 8,000 when the true population average is actually INR 8,000. The estimated gap is 0.9382 thousand (about INR 938); it is statistically significant, but modest relative to the average cost and should be considered alongside budgeting materiality.

The required sample size for a margin of error of 0.5 thousand is 1,293, compared with 800 available records, a shortfall of 493. The planning team would need more observations to meet the stated precision target under this estimate of variability.

Gradient descent's slope gap is 0.0000. With the stated learning rate, full-batch updates and 500 iterations, it converges to the same least-squares line as the closed-form calculation.

The raw residual spread ratio is 3.7525, which indicates a funnel: residual spread is much larger in the upper fitted-value half. After the log-log transformation the ratio is 1.0396, so the spread is nearly even and the transformation largely removes the funnel.

Using the mechanical absolute-coefficient cutoff of 0.05 in the multiple regression, the qualifying predictors are ln(X4), X2, X3 and X6. The adjusted R² rises from 0.6561 for the log-log single-predictor model to 0.7403 for the multiple regression, suggesting the additional predictors improve in-sample explanatory fit; this does not establish causal importance.

## Part 2 findings

After 200 updates, the losses from lowest to highest are η = 1.0 (0.3703), η = 0.1 (0.3776), and η = 0.01 (0.5091). The smaller learning rate is slower to reduce loss over this fixed budget of updates; η = 1.0 performs best among the three rates tested, though that does not guarantee that still larger rates would remain stable.

The three largest absolute logistic weights are X3 (0.9013), X1 (−0.5554), and X2 (0.5505). X3 has the largest absolute weight; its odds ratio is 2.4629, meaning a one-training-standard-deviation increase in copay burden is associated with about 2.46 times the modelled odds of a no-show, holding the other standardised predictors fixed.

The all-negative baseline has accuracy 0.7875, close to the logistic model's 0.8042 accuracy at τ = 0.5. Accuracy alone is weak here because only 51 of 240 test records are positive, so predicting every record as a non-no-show already scores well. Lowering τ from 0.5 to 0.3 raises recall from 0.4706 to 0.7451, while precision falls from 0.5455 to 0.4634.

At τ = 0.3, logistic regression has F1 = 0.5714 and recall = 0.7451. KNN has F1 = 0.3696 and recall = 0.3333, so logistic regression is stronger on both measures for this split.

I would use the logistic model with the cost-selected threshold τ = 0.15 as the candidate reminder policy: its test cost is INR 6,050, compared with INR 11,800 at τ = 0.5 and INR 20,400 if nobody is flagged. The threshold is selected using training rows so that the test set remains an honest evaluation set rather than influencing the decision rule.

## Part 3 findings

The successive WCSS drops are 659.8282 (K2→K3), 108.0739 (K3→K4), 171.0575 (K4→K5), and 45.5466 (K5→K6). The elbow rule selects K = 3 because the K3→K4 drop is less than half the preceding drop. The later drops do not shrink monotonically; the pinned starting rows give a reproducible solution, not necessarily the global optimum.

- **Segment A — Established, low-friction users:** 322 records; mean missed/rescheduled visits 1.9503, copay burden 0.2250 and outcome rate 0.0186. Keep the experience simple and use routine appointment reminders.
- **Segment B — Frequent reschedulers:** 255 records; mean missed/rescheduled visits 6.0706, average bill 28.7192 thousand and outcome rate 0.2353. Offer easy rescheduling and timely reminder follow-ups.
- **Segment C — Newer, high-copay users:** 223 records; mean tenure 14.3892 months, copay burden 0.7118, adherence score 38.5215 and outcome rate 0.4664. Prioritise clear cost communication and additional appointment support.

Segment C has the highest no-show outcome rate, 0.4664. These are service-use profiles, not medical categories.

At the largest-gap cut, single linkage gives four clusters: {E1}, {E2,E3,E6}, {E4}, {E5}. Complete linkage gives two clusters: {E1,E4,E5} and {E2,E3,E6}. Single linkage joins clusters when any pair of members is close, so a chain of short links can connect E2, E3 and E6 early. Complete linkage uses the farthest cross-cluster pair, delaying a merge until all members are relatively close and therefore producing a different cut.

After standardisation, the feature mean vector is approximately all zeros. PC1 is driven positively by X1 (+0.5317) and X5 (+0.4418), and negatively by X3 (−0.5187), X2 (−0.4100) and X4 (−0.2916). Three components retain at least 80% of variance (cumulative explained variance 0.8616). Mean PC1 scores are A = 1.7773, B = −0.7831 and C = −1.6709, showing A at the positive end of this component and C at the negative end, with B between them.

## Part 4 findings

S4 (X1 ≤ 24) is the best split: it has the largest information gain (0.1252) and Gini gain (0.0606) among the four candidates. A tree grown until every leaf is pure can memorise noise and small quirks in the training data, which harms generalisation. A maximum-depth limit, a minimum-leaf-size rule or post-pruning can stop excessive growth.

Of the five test-set predictions, logistic regression has the strongest F1 (0.5714) with recall 0.7451. The stump has slightly higher recall (0.7647) but lower F1 (0.5532); hard voting gives F1 0.5344 and recall 0.6863, while soft voting gives F1 0.5507 and recall 0.7451. Combining the models did not improve F1 over logistic regression alone on this split.

After round 1 of AdaBoost, the two misclassified points (3 and 7) each have normalised weight 0.2500, for a combined weight of 0.5000. Round 2 gives h2 more influence because its weighted error is lower (0.1667 versus 0.2500), producing a larger alpha (0.8047 versus 0.5493).

**Model A — CNN output shapes and parameter counts**

| Layer | Output shape | Parameters |
|---|---:|---:|
| Conv1 (5×5, 6 filters, padding 2) | 28×28×6 | 156 |
| Max-pool 1 (2×2) | 14×14×6 | 0 |
| Conv2 (5×5, 16 filters, no padding) | 10×10×16 | 2,416 |
| Max-pool 2 (2×2) | 5×5×16 | 0 |
| Flatten | 400 | 0 |
| Dense hidden (32) | 32 | 12,832 |
| Dense output (10) | 10 | 330 |

The MLP alternative has 13,002 parameters and the CNN has 15,734. For the separate 128×128×3 scaling comparison, the fully connected layer has 50,332,672 parameters, while the 3×3 convolutional layer has 1,792.

**Model B — RNN shapes and design.** W_xh has shape 128×64, W_hh is 128×128, and W_hy is 16×128; including biases, the total is 26,768 parameters. The parameter count does not grow with sequence length because the same weight matrices are reused at every time step. For 10 scan classes, I would use ReLU in hidden layers and softmax with categorical cross-entropy at the output. In a very deep plain network, repeated gradients can shrink as they travel backwards, leaving early layers with little learning signal; skip connections provide a shorter path for information and gradients and help preserve learning in earlier layers.

## Model-choice memo

**To: Head of Scheduling Operations**

I recommend logistic regression as the first structured-data candidate, using the cost-selected alert threshold of 0.15 for evaluation in a controlled pilot. On the prescribed test split it achieves F1 = 0.5714 at τ = 0.3, compared with KNN F1 = 0.3696, and the cost rule gives INR 6,050 at τ = 0.15. The same logistic model costs INR 11,800 at τ = 0.5, while flagging nobody costs INR 20,400. The threshold was chosen on training data, not tuned on the test set. This remains a candidate rather than a production-ready clinical or operational rule: the evidence is synthetic and the test set contains only 51 positive outcomes. I would not build Model A or Model B before validating the structured model; first test the reminder workflow and its cost assumptions with representative operational data, then revisit image or referral-note routing as separate projects.

## Limitations

- The 800-row dataset is synthetic; it is not PharmEasy data and cannot establish real patient or business behaviour.
- Evaluation uses one prescribed stratified 70/30 split, so results may change on another sample or time period.
- The test set contains only 51 positive no-show records, making positive-class metrics uncertain.
- The cost threshold depends on the assumed INR 50 false-alarm and INR 400 missed-no-show costs; actual operational costs may differ.
- The segmentation and model results are descriptive/predictive associations, not causal findings or validated clinical tools.
- The neural-network sections count parameters and shapes only; neither architecture is trained or evaluated.
