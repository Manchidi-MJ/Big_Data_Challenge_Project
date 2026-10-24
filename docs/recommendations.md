# Retention Recommendations (Objective 6)

> Fill the bracketed figures with your actual results after running the
> pipeline on the real Kaggle data. The structure links each recommendation
> back to an EDA / RFM / model finding.

Connecting the behaviour analysis (Stage 2), the RFM segmentation (Stage 3)
and the model's feature importances (Stage 4) gives a practical retention plan.

## 1. Resolve complaints fast
**Finding:** customers who complained churn at **[X%]** vs **[Y%]** for those
who did not; `Complain` is among the top model features.
**Action:** prioritise complaint resolution and follow up proactively; flag any
complaint as an immediate churn-risk signal.

## 2. Protect new / low-tenure customers
**Finding:** churn concentrates in low `Tenure` customers; tenure is the
strongest single predictor.
**Action:** structured onboarding and early-life incentives in the first few
months, when risk is highest.

## 3. Win back the "At Risk" and "Lost" RFM segments
**Finding:** these segments show the highest churn rates (**[A%]**, **[B%]**),
driven by high recency (long since last order) and low frequency.
**Action:** targeted reactivation campaigns (personalised offers, reminders) for
these segments specifically, rather than spending uniformly across all
customers.

## 4. Reward and retain "Champions" / "Loyal"
**Finding:** low churn but high value.
**Action:** loyalty perks and early access to defend the most valuable base
cheaply.

## 5. Use cashback / incentives where they move the needle
**Finding:** higher `CashbackAmount` associates with lower churn.
**Action:** test targeted cashback for at-risk, price-sensitive segments and
measure uplift — but avoid subsidising customers who would stay anyway.

## 6. Act on low satisfaction scores
**Finding:** churn rises as `SatisfactionScore` falls.
**Action:** trigger outreach when satisfaction drops below a threshold.

## Prioritisation
Focus spend where predicted churn probability **and** customer value are both
high (At-Risk / Lost segments with non-trivial Monetary scores). This directly
answers the problem statement: targeting retention at the right customers
instead of spreading effort thinly.
