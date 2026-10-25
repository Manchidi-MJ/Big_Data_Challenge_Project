# References

All sources below were consulted during the project. Each is cited in the final
report. URLs were checked and were reachable at the time of writing
(October 2026). Publication details were verified against the publisher or a
canonical index; where a detail could not be fully confirmed it is marked.

## Customer churn and retention

1. Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., & Mason, C. H. (2006).
   Defection Detection: Measuring and Understanding the Predictive Accuracy of
   Customer Churn Models. *Journal of Marketing Research*, 43(2), 204–211.
   DOI: 10.1509/jmkr.43.2.204.

2. Reichheld, F. F., & Sasser, W. E. (1990). Zero Defections: Quality Comes to
   Services. *Harvard Business Review*, 68(5), 105–111.
   (Frequently cited source for the economics of customer retention; see the
   note in §3 of the report, where the "retention is cheaper than acquisition"
   claim is stated cautiously rather than as an established universal law.)

## RFM analysis and customer segmentation

3. Hughes, A. M. (1994). *Strategic Database Marketing: The Masterplan for
   Starting and Managing a Profitable, Customer-Based Marketing Program.*
   Chicago: Probus Publishing. (Origin of the five-bin RFM scoring approach.)
   Record: <https://archive.org/details/strategicdatabas00hugh>

4. Kohavi, R., & Parekh, R. (2004). Visualizing RFM Segmentation. In
   *Proceedings of the 2004 SIAM International Conference on Data Mining
   (SDM)*, 391–399. SIAM. DOI: 10.1137/1.9781611972740.37.
   PDF: <http://ai.stanford.edu/~ronnyk/RFMSDM2004.pdf>
   (Describes the common Arthur Hughes method of binning each RFM attribute into
   five equal-frequency bins.)

## Machine learning: models

5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
   DOI: 10.1023/A:1010933404324.

6. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic
   Regression* (3rd ed.). Hoboken, NJ: Wiley.
   DOI: 10.1002/9781118548387.

7. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
   O., … Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.
   <https://jmlr.org/papers/v12/pedregosa11a.html>

## Model evaluation and class imbalance

8. Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall
   and ROC Curves. In *Proceedings of the 23rd International Conference on
   Machine Learning (ICML)*, 233–240. DOI: 10.1145/1143844.1143874.

9. Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More
   Informative than the ROC Plot When Evaluating Binary Classifiers on
   Imbalanced Datasets. *PLoS ONE*, 10(3), e0118432.
   DOI: 10.1371/journal.pone.0118432.

## Data leakage

10. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in
    Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on
    Knowledge Discovery from Data (TKDD)*, 6(4), Article 15, 1–21.
    DOI: 10.1145/2382577.2382579.
    <https://dl.acm.org/doi/10.1145/2382577.2382579>

## Distributed and reproducible data processing

11. Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., …
    Stoica, I. (2016). Apache Spark: A Unified Engine for Big Data Processing.
    *Communications of the ACM*, 59(11), 56–65. DOI: 10.1145/2934664.
    <https://dl.acm.org/doi/10.1145/2934664>

12. Apache Software Foundation. (2024). *PySpark Documentation* (Spark 3.5.x).
    <https://spark.apache.org/docs/3.5.1/api/python/index.html>
    (Official documentation; consulted for the DataFrame API used in the
    cleaning pipeline.)

## Dataset

13. Maaz, M. (2025). *E-Commerce Customer Churn 2025 – 10K* [Data set]. Kaggle.
    Licence: MIT. Retrieved 2026-01.
    <https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k>

---

### Note on citation completeness

Items 1, 2, 5, 6, 8 and 9 are long-established, widely indexed works whose
bibliographic details (authors, venue, year, DOI) are reproduced from standard
citation records. Items 3, 4, 10, 11, 12 and 13 were additionally confirmed via
a reachable URL during this project. No quotations longer than a short phrase
are reproduced from any source, and all findings reported in this project are
the team's own computed results, not values copied from the literature.
