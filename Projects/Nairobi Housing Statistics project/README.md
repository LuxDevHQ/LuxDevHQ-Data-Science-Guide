# Nairobi Rentals — Statistics Practice Guide

A guided set of practice questions that takes you from describing data, through measuring spread, to testing hypotheses with both parametric and non-parametric methods, using one dataset: `nairobi_rentals_reemio.csv`.

## 1. About the dataset

The file holds 1,486 residential rental listings across 110 neighbourhoods in Nairobi, Kiambu, Kajiado and Machakos counties, with 29 columns and no missing values. Listings were collected from Reemio Listings (listings.reemioltd.com) on 19 September 2026 and cover listing dates from December 2025 to September 2026.

Some columns are real and some are simulated for teaching. Be honest about this in any report you write: conclusions drawn from simulated columns say nothing about the actual Nairobi market.

| Column | Type | Source | Notes |
|---|---|---|---|
| `listing_id` | ID | generated | Unique row key |
| `listing_date`, `listing_weekday` | date, category | real | When the unit was listed |
| `neighbourhood`, `sub_county`, `county` | category | real | Location |
| `market_segment` | category | real | Reemio's tag: Affordable, Student Friendly, Family Friendly, Mid-Range, Luxury |
| `property_category`, `unit_type` | category | real | Apartment, Townhouse, etc.; Studio to 6 Bedroom |
| `bedrooms`, `bathrooms`, `balconies`, `floor_level` | discrete | real | `bedrooms = 0` means studio/bedsitter; `floor_level = 0` means ground |
| `building_id` | ID | real (anonymised) | Units sharing an ID are in the same building |
| `rent_kes_2026` | continuous | real | Monthly asking rent, KES |
| `deposit_kes` | continuous | real | Deposit required, KES |
| `distance_to_cbd_km` | continuous | derived | From the neighbourhood's average coordinates |
| `size_sqm` | continuous | simulated | Normal within each bedroom class |
| `building_age_years` | discrete | simulated | Right-skewed |
| `furnished`, `parking`, `gym`, `borehole` | Yes/No | simulated | First three are linked to real rent; `borehole` is random |
| `peak_commute_min` | continuous | simulated | Approximately normal |
| `rent_kes_2025` | continuous | simulated | The same unit's rent 12 months earlier (for paired tests) |
| `days_on_market` | discrete | simulated | Right-skewed |
| `building_occupancy_pct` | continuous | simulated | Left-skewed |
| `tenant_satisfaction_10` | continuous | simulated | Left-skewed, score out of 10 |
| `listed_by` | category | simulated | Random; has no relationship with rent |

## 2. Getting started

```python
import pandas as pd, numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

df = pd.read_csv("nairobi_rentals_reemio.csv", parse_dates=["listing_date"])
df.info()
df.describe()
```

Before any analysis, classify every column as nominal, ordinal, discrete or continuous. The choice of summary statistic and test depends on it.

Throughout, use a significance level of α = 0.05 unless told otherwise, and for every test write down: the hypotheses (H₀ and H₁), the test chosen and why, the assumptions and how you checked them, the test statistic and p-value, and a conclusion in plain language a landlord or tenant would understand.

## 3. Part A — Measures of central tendency

**Key idea.** In this guide "mean" always refers to the ordinary arithmetic mean: add up the values and divide by how many there are. The mean uses every value and is pulled towards extreme ones; the median depends only on the middle of the ordered data; the mode is the most frequent value and is the only one that works for categories. In a right-skewed distribution mean > median; in a left-skewed one mean < median.

**Practice questions**

A1. Compute the mean, median and mode of `rent_kes_2026`. Which is largest, and what does that tell you about the shape of the distribution?

A2. A tenant asks "what does a typical house cost to rent?" Which of the three measures would you quote, and why?

A3. Compute the mean by hand in Python with `df["rent_kes_2026"].sum() / len(df)` and confirm it matches `.mean()`. How many listings rent for less than the mean? Is it about half, or many more than half? Why?

A4. Find the mean and median rent for each value of `bedrooms`. For which bedroom class is the gap between mean and median widest in percentage terms?

A5. Find the mode of `bedrooms`, `market_segment` and `neighbourhood`. Why is the mean meaningless for the last two?

A6. Remove the five most expensive listings and recompute the mean and median rent. Which moved more? What property of the median does this demonstrate?

A7. Compare mean and median for `building_occupancy_pct` and for `peak_commute_min`. Without plotting, predict the shape of each distribution, then check with a histogram.

A8. Compute the mean rent per county, then the mean of those four county means. Why does it differ from the overall mean? (Hint: weighted mean.)

## 4. Part B — Measures of dispersion

**Key idea.** Two datasets can share a mean and look nothing alike. Range, interquartile range (IQR), variance, standard deviation and the coefficient of variation (CV = standard deviation ÷ mean) each describe spread. The IQR looks only at the middle half of the data, so extreme values do not affect it; the others are all affected by extremes.

**Practice questions**

B1. For `rent_kes_2026` compute the range, IQR, variance and standard deviation. Why is the variance such an awkward number to interpret?

B2. Compute the CV of rent for each county. Which county has the most variable rents relative to its average? Why is CV fairer here than comparing standard deviations directly?

B3. Use the 1.5 × IQR rule to find the upper fence for rent. How many listings fall above it? Look at them: are they errors or genuine luxury units? Should they be deleted?

B4. Draw a boxplot of rent by `bedrooms`, then repeat with rent on a log scale. What becomes visible on the log scale that was hidden before?

B5. Compute the five-number summary of `days_on_market`. How far is the maximum from Q3 compared with the minimum from Q1?

B6. Compute skewness (`.skew()`) for `rent_kes_2026`, `days_on_market`, `building_occupancy_pct`, `tenant_satisfaction_10` and `peak_commute_min`. Sort the columns into right-skewed, left-skewed and roughly symmetric.

B7. Convert rent to z-scores. How many listings have |z| > 3? Compare with the count from the IQR rule in B3 and explain why the two disagree.

B8. Compare the standard deviation of `size_sqm` for the whole dataset with the standard deviation within 2-bedroom units only. Why is the second so much smaller?

## 5. Part C — Distributions and normality

**Key idea.** Parametric tests assume the data (or the sampling distribution of the mean) are roughly normal. Check with a histogram, a Q–Q plot and a formal test such as Shapiro–Wilk, in that order of importance.

**Practice questions**

C1. Plot histograms of `peak_commute_min`, `rent_kes_2026` and `building_occupancy_pct` side by side. Label each as normal, right-skewed or left-skewed.

C2. Test the empirical rule on `peak_commute_min`: what share of values lie within 1, 2 and 3 standard deviations of the mean? Repeat for rent and explain the difference.

C3. Run Shapiro–Wilk on a random sample of 500 values from `peak_commute_min` and from `rent_kes_2026`. Why sample rather than use all 1,486 rows?

C4. Apply a natural log to rent and re-plot. Is log-rent normal? Does Shapiro–Wilk agree with your eyes? What does this teach you about formal normality tests on large samples?

C5. `size_sqm` looks skewed overall but is normal within 2-bedroom units. Explain how mixing several normal groups produces a non-normal whole.

C6. Central limit theorem: draw 1,000 random samples of size 40 from rent, store each sample mean, and plot the 1,000 means. Describe the shape. Why does this justify using a t-test on skewed rent data when n is large?

C7. Build a 95% confidence interval for the mean rent. Explain in one sentence what the interval means.

## 6. Part D — Parametric hypothesis tests

**One-sample t-test** — compares one group's mean with a claimed value.

D1. An agent claims the average 1-bedroom in Utawala rents for KES 18,000. Test the claim.

D2. A county planner says the average peak commute is 50 minutes. Test it using `peak_commute_min`.

D3. Test whether the mean rent of 2-bedroom units in Ruaka differs from KES 30,000. With only 29 listings, which assumption matters most?

**Independent two-sample t-test** — compares the means of two unrelated groups.

D4. Among 2-bedroom units, do units with parking rent for more than units without? Run Levene's test first, then decide between Student's and Welch's t-test.

D5. Compare mean rent for furnished and unfurnished units. The groups are very unequal in size (about 10% are furnished). Does that invalidate the test?

D6. Repeat D4 on log-rent. Do the conclusions change? Which version better satisfies the assumptions?

**Paired t-test** — compares two measurements on the same units.

D7. Did rents rise between 2025 and 2026? Use `rent_kes_2025` and `rent_kes_2026`. Why is an independent t-test wrong here, even though it would run without error?

D8. Compute the differences first and run a one-sample t-test of the differences against zero. Confirm you get the same t statistic as in D7.

D9. About 30% of units show no change at all. Plot the differences. Are they normal? Does the paired t-test still hold with n = 1,486?

**One-way ANOVA** — compares the means of three or more groups.

D10. Among 2-bedroom units, does mean rent differ across the four counties? If ANOVA is significant, look at the four county means to see which counties stand apart. (Optional: run Tukey's HSD, `st.tukey_hsd`, to confirm which pairs differ.)

D11. Does mean rent differ by `listed_by`? Interpret a non-significant result correctly: does it prove the means are equal?

D12. Test whether mean rent differs by `market_segment`, then check Levene's test. What should you do when group variances are this unequal?

**Correlation**

D13. Compute Pearson's r between `distance_to_cbd_km` and `rent_kes_2026`. Interpret the sign and the strength. Does it prove that distance causes lower rent?

## 7. Part E — Non-parametric hypothesis tests

**Key idea.** Non-parametric tests work on ranks or counts, so they do not need normality and are not thrown off by outliers. The price is slightly less power when the data really are normal. Use them for skewed data with small samples, ordinal data, or when outliers are genuine and cannot be removed.

| Parametric test | Non-parametric counterpart |
|---|---|
| One-sample t-test | Wilcoxon signed-rank (one sample) |
| Independent t-test | Mann–Whitney U |
| Paired t-test | Wilcoxon signed-rank (paired) |
| One-way ANOVA | Kruskal–Wallis H |
| Pearson correlation | Spearman rank correlation |
| — (categorical data) | Chi-square tests |

**Practice questions**

E1. Repeat D1 (Utawala 1-bedroom vs KES 18,000) with the Wilcoxon signed-rank test. Do the two tests agree?

E2. Repeat D4 (parking vs no parking, 2-bedroom) with Mann–Whitney U. State the hypotheses carefully: what exactly is Mann–Whitney comparing?

E3. Do tenants in buildings with a gym report higher `tenant_satisfaction_10`? Satisfaction is left-skewed and bounded at 10, so choose your test accordingly.

E4. Repeat D7 (2025 vs 2026 rent) with the paired Wilcoxon signed-rank test. How does the test treat the units whose rent did not change?

E5. Repeat D10 (2-bedroom rent by county) with Kruskal–Wallis. Does it agree with the ANOVA? Looking at the county medians, which county or counties seem to drive the result?

E6. Compute Spearman's ρ between distance to CBD and rent, and compare it with Pearson's r from D13. Why is ρ larger in magnitude here?

E7. Chi-square test of independence: is `furnished` independent of `market_segment`? Check the expected counts. Is any cell below 5, and what would you do about it?

E8. Chi-square test of independence: is `parking` related to `county`?

E9. Chi-square goodness of fit: are listings spread evenly across the seven days of the week? What real-world behaviour might explain the result?

E10. For each of D1, D4, D7 and D10, write one sentence saying which version (parametric or non-parametric) you would report to a client and why.

## 8. Choosing a test — quick guide

Start with the outcome variable. If it is categorical, you are counting, so use chi-square (goodness of fit for one variable, independence for two). If it is numeric, ask how many groups you are comparing and whether the measurements are paired.

| Situation | Data roughly normal, or n large | Skewed with small n, ordinal, or heavy outliers |
|---|---|---|
| One group vs a claimed value | One-sample t | Wilcoxon signed-rank |
| Two independent groups | Welch's t | Mann–Whitney U |
| Two measurements on the same units | Paired t | Wilcoxon signed-rank (paired) |
| Three or more independent groups | One-way ANOVA | Kruskal–Wallis |
| Two numeric variables | Pearson r | Spearman ρ |

Useful `scipy.stats` functions: `ttest_1samp`, `ttest_ind(equal_var=False)`, `ttest_rel`, `f_oneway`, `tukey_hsd`, `levene`, `shapiro`, `wilcoxon`, `mannwhitneyu`, `kruskal`, `spearmanr`, `pearsonr`, `chi2_contingency`, `chisquare`.

## 9. Common mistakes to avoid

Reporting the mean rent as "typical" when the distribution is strongly skewed. Deleting outliers simply because they are inconvenient; the KES 240K–510K Westlands units are genuine. Saying "we accept H₀" or "the means are equal" after a non-significant result; you have only failed to find evidence of a difference. Treating a tiny p-value as a large effect: with 1,486 rows almost anything is significant, so always report the size of the difference too. Using an independent test on paired data. Reading correlation as causation.

## 10. Answer key (selected checkpoints)

Use these to check your work. Small differences from rounding or random sampling are fine.

**Part A.** Rent: mean ≈ 33,753; median = 25,000; mode = 20,000. Mean > median > mode signals right skew. A3: far more than half of listings sit below the mean, because a few very expensive units pull it up. Mode of `bedrooms` is 2. Median rent by bedrooms: studio 15,000; 1-bed 18,000; 2-bed 30,000; 3-bed 65,000. Occupancy: mean 87.7 < median 89.9 (left-skewed). Commute: mean 50.05 ≈ median 50 (symmetric).

**Part B.** Rent: range 6,000–510,000; Q1 = 18,000; Q3 = 37,000; IQR = 19,000; sd ≈ 31,488; CV ≈ 0.93. Upper fence = 65,500, with 119 listings above it. Rent skewness ≈ 6.0. Skewness of other columns: `days_on_market` ≈ 3.0, `building_occupancy_pct` ≈ −1.2, `tenant_satisfaction_10` ≈ −0.7, `peak_commute_min` ≈ 0.1. Nairobi county has the highest CV (sd 43,001 on a mean of 41,168).

**Part C.** Empirical rule on commute: about 67%, 95% and 99.7%. Shapiro–Wilk on a 500-row sample: commute p ≈ 0.5 (no evidence against normality); rent p < 0.001. Log-rent is much closer to normal but still fails Shapiro–Wilk. 95% CI for mean rent: roughly 32,150 to 35,355.

**Part D.** D1: n = 51, mean ≈ 18,882, t ≈ 1.39, p ≈ 0.17, fail to reject. D2: t ≈ 0.19, p ≈ 0.85, fail to reject. D3: n = 29, mean ≈ 40,172, p ≈ 0.001, reject. D4: parking mean ≈ 41,418 (n = 445) vs 25,237 (n = 252); Levene p < 0.001 so use Welch; t ≈ 9.9, p < 0.001. D7: mean rise ≈ KES 1,357, t ≈ 26.7, p < 0.001. D10: F ≈ 10.0, p < 0.001; Tukey shows only Kiambu vs Nairobi differs significantly (difference ≈ 11,270). D11: F ≈ 0.40, p ≈ 0.67, fail to reject. D13: r ≈ −0.38.

**Part E.** E1: Wilcoxon p ≈ 0.23, agrees with the t-test. E2: Mann–Whitney p < 0.001. E3: median 8.2 (gym) vs 7.9 (no gym), Mann–Whitney p ≈ 0.009. E4: p < 0.001; zero differences are dropped by default. E5: H ≈ 24.3, p < 0.001, agreeing with the ANOVA; Nairobi and Machakos have the highest 2-bedroom rents. E6: ρ ≈ −0.47, stronger than r because the relationship is monotonic but not linear and ranks ignore the outliers. E7: χ² ≈ 83.7, df = 4, p < 0.001; the Luxury row is small (16 units), so check expected counts. E8: χ² ≈ 9.58, df = 3, p ≈ 0.022. E9: χ² ≈ 352, p < 0.001; listings cluster on weekdays (Tuesday 334, Sunday 17), consistent with agents uploading during working days.

## 11. What to submit

Submit **one link to a public GitHub repository**. Do not send files by email or chat.

**Naming the repository.** Name it after the subject, for example `nairobi-rental-housing` or `nairobi-rentals-statistics`. **Do not put the word "project" in the name**: `Nairobi Rental housing project` should simply be `nairobi-rental-housing`. Every repository is already a project, so the word adds nothing, and employers browsing your GitHub profile read names like `xyz-project` as coursework. Use lowercase words joined by hyphens, with no spaces.

**Folder structure.** Your repository should look like this:

```
nairobi-rental-housing/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── nairobi_rentals_reemio.csv
├── notebooks/
│   └── nairobi_rentals_statistics.ipynb
└── images/
    └── (charts saved from the notebook)
```

**README.md (your own, not this file).** It should tell a visitor, in under two minutes: what the repository is about, where the data came from and which columns are simulated, how to run the notebook (`pip install -r requirements.txt`), your three to five most important findings in plain language with one or two charts, and the tools you used.

**The notebook.** One detailed `.ipynb` that runs from top to bottom without errors (use *Restart & Run All* before you push). It must follow the parts of this guide in order (central tendency, dispersion, distributions, parametric tests, non-parametric tests) with a markdown heading for each part. Markdown cells matter as much as code cells: before each code cell, say what you are about to do and why; after each result, interpret it in plain language. For every hypothesis test, write the hypotheses, the test you chose and why, the assumption checks, the result, and the conclusion. Label every chart with a title and axis labels. A notebook with code and no explanation will be returned.

**Checklist before you submit**

- [ ] Repository is public and the link opens in a private/incognito browser window
- [ ] Repository name has no "project" in it, no spaces, all lowercase with hyphens
- [ ] Folder structure matches the one above; the notebook loads the CSV with a relative path (`../data/nairobi_rentals_reemio.csv`)
- [ ] Notebook runs top to bottom after *Restart & Run All*, and outputs are visible on GitHub
- [ ] Every test has hypotheses, assumption checks and a plain-language conclusion in markdown
- [ ] README has findings and at least one chart
- [ ] Several meaningful commits, not a single "final upload"

## 12. Extension tasks

Write a two-page market brief for a first-time renter choosing between Ruaka, Utawala and Syokimau, using only descriptive statistics and one test. Investigate whether floor level is related to rent in buildings with six or more floors. Group listings by `building_id` and discuss why units in the same building are not truly independent observations, and what that means for every test above.
