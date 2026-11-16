Domain: Healthcare

# PharmEasy Healthcare Decision Intelligence

## How to run

Run these commands in order from a clean clone, from the repository root:

```bash
python -m venv .venv
# Activate the environment:
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The project uses Python 3.10+ and the pinned NumPy dependency. The data-generation script uses the Python standard library.

## Part 1 findings

The analysis uses 800 synthetic healthcare records. Mean care-plan cost is 8.9382 thousand INR, with sample standard deviation 9.1723 and standard error 0.3243. The 95% confidence interval is 8.3026–9.5738 thousand INR. The test against a mean of 8.0 gives z = 2.8930 (reject at the 5% level). The estimated simple-regression slope is 0.6701 (R² = 0.5196); gradient descent agrees with the closed-form slope to the reported precision.

## Part 2 findings

The stratified split contains 560 training rows (119 positive outcomes) and 240 test rows (51 positive outcomes). Logistic-regression test BCE is 0.3818. At threshold 0.5, F1 is 0.5053; at threshold 0.3, F1 is 0.5714. Five-fold cross-validation selects K = 5 for KNN. The cost-minimizing training threshold is 0.15; reported test cost is INR 6,050, compared with INR 11,800 at threshold 0.5.

## Part 3 findings

The selected elbow K is 3, with segment sizes A = 322, B = 255, and C = 223. At the largest-gap cut, hierarchical cluster counts are single = 4, complete = 2, and average = 2. The first three PCA explained-variance ratios are 0.5227, 0.2366, and 0.1023; three components exceed 80% cumulative explained variance.

## Part 4 findings

The best decision-tree split is S4 under both information gain and Gini gain. The two-round AdaBoost errors are 0.2500 and 0.1667, with alphas 0.5493 and 0.8047; ensemble accuracy is 0.7500. The architecture calculations report 13,002 MLP parameters, 15,734 CNN parameters, 50,332,672 fully connected-layer parameters, 1,792 convolution-layer parameters, and 26,768 RNN parameters.

## Model-choice memo

Logistic regression is a practical baseline for the binary no-show outcome because it is interpretable, produces probabilities, and supports an explicit threshold decision tied to the stated cost assumptions. KNN is retained as a comparison model, with K selected by the prescribed cross-validation procedure. The chosen threshold should be treated as a decision aid rather than a clinical rule: it depends on the assumed relative costs and the synthetic data distribution.

## Limitations

- The dataset is synthetic and cannot establish real-world healthcare or patient behaviour.
- The evaluation uses a single prescribed train/test split; results may vary across alternative splits.
- The test set contains only 51 positive outcomes, so positive-class metrics have sampling uncertainty.
- The threshold recommendation depends on the cost assumptions encoded in the assignment.
- The models are educational analyses, not validated clinical tools and must not be used for patient-level decisions.
