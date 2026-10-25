# Retention Recommendations (Objective 6)

> Fill the bracketed figures with your actual results after running the
> pipeline on the real Kaggle data. The structure links each recommendation
> back to an EDA / RFM / model finding.

Connecting the behaviour analysis (Stage 2), the RFM segmentation (Stage 3)
and the model's feature importances (Stage 4) gives a practical retention plan.

## 1. Re-engage lapsing customers quickly
**Finding:** churned customers have a much higher `days_since_last_purchase`
(**[X]** days vs **[Y]** for retained); recency is a top model feature.
**Action:** trigger win-back campaigns (reminders, personalised offers) once a
customer crosses a recency threshold, before they fully lapse.

## 2. Grow order frequency and spend
**Finding:** retained customers have higher `total_orders` and
`total_spent_usd`; low-frequency, low-spend customers churn more.
**Action:** encourage repeat purchase (subscriptions, bundles, loyalty points)
for low-frequency customers to move them up the RFM ladder.

## 3. Convert and retain with premium membership
**Finding:** `is_premium_member` customers churn at **[A%]** vs **[B%]** for
regular customers.
**Action:** promote membership to high-value non-members and protect the
premium base with exclusive perks.

## 4. Win back the "At Risk" and "Lost" RFM segments
**Finding:** these segments show the highest churn rates (**[C%]**, **[D%]**),
driven by high recency and low frequency.
**Action:** targeted reactivation spend on these segments specifically, rather
than spreading retention budget evenly across all customers.

## 5. Reward "Champions" / "Loyal"
**Finding:** low churn but high value.
**Action:** loyalty perks and early access to defend the most valuable base
cheaply.

## 6. Tailor by device / category / country
**Finding:** churn rate varies by `device_type`, `favorite_category` and
`country` (highest for **[segment]**).
**Action:** investigate friction in the highest-churn segments (e.g. a poor
mobile experience, a weak category) and prioritise fixes there.

## Prioritisation
Focus spend where predicted churn probability **and** customer value
(Monetary) are both high — At-Risk / Lost segments with meaningful spend. This
directly answers the problem statement: target retention at the right
customers instead of spreading effort thinly.
