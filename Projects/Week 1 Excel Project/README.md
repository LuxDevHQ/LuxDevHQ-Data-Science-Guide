# Jumia Product Performance Dashboard

## An Excel project guide for pricing, discounts, and customer reviews

## 1. Project objective

Build a professional, interactive Microsoft Excel dashboard that turns the supplied Jumia product data into useful pricing, promotion, and customer-engagement insights. By the end of the project, you should be able to explain:

- whether larger discounts are associated with more reviews;
- whether highly rated products attract stronger engagement;
- whether price and rating move together;
- which products perform best based on ratings and reviews; and
- which products may need a different pricing or marketing strategy.

This is an analysis project, not only a chart-building exercise. Preserve the raw data, make every cleaning decision traceable, validate calculations, and connect each recommendation to evidence in the workbook.

## 2. Business context

Jumia and its sellers need to understand how price, promotions, and customer feedback influence product performance. In this project, the number of reviews is used as an **engagement proxy** because the dataset does not include units sold or revenue. Do not describe reviews as sales or claim that a relationship proves causation.

## 3. Starter dataset

Use [`Excel_jumia_dataset.csv`](./Excel_jumia_dataset.csv). It contains these source fields:

| Source field | Meaning | Expected cleaned type |
|---|---|---|
| `Product` | Product name | Text |
| `Current price` | Current selling price in Kenyan shillings (KSh) | Number/currency |
| `old price` | Price before discount in KSh | Number/currency |
| `Discount` | Advertised percentage discount | Percentage |
| `Review` | Number of customer reviews | Whole number |
| `Ratingd` | Average rating out of 5 | Decimal number |

> **Important data-quality observations:** the source header `Ratingd` is misspelled, some review and rating cells are blank, review counts appear with negative signs, and at least one price is expressed as a range. The file also contains repeated rows. Investigate and document these issues instead of silently changing them.

## 4. Required submission structure

Create one workbook, for example `jumia_product_dashboard.xlsx`, containing at least these worksheets:

1. **Raw_Data** — an untouched copy of the CSV;
2. **Cleaned_Data** — cleaned fields, derived columns, and an Excel Table;
3. **Analysis** — descriptive statistics, correlation checks, and ranked product tables;
4. **Pivot_Tables** — all PivotTables that support the visuals;
5. **Dashboard** — KPIs, charts, slicers, and short insights; and
6. **Data_Dictionary** — field definitions, assumptions, thresholds, and a cleaning log.

Do not overwrite `Raw_Data`. Convert the cleaned range to an Excel Table with **Ctrl+T**, confirm that it has headers, and give it a clear name such as `tblProducts`.

## 5. Step 1 — Import, audit, and clean the data

### 5.1 Import safely

1. Open a blank workbook.
2. Select **Data > Get Data > From Text/CSV**.
3. Choose `Excel_jumia_dataset.csv` and verify that the comma delimiter is detected.
4. Load the source into `Raw_Data`.
5. Duplicate the query or sheet before cleaning and load the result into `Cleaned_Data`.

Power Query is recommended because its steps are repeatable. Excel formulas are also acceptable if the raw sheet remains unchanged.

### 5.2 Create a data-quality audit

Before correcting anything, record:

- source row count and column count;
- blank count by column;
- duplicate count and the fields used to identify duplicates;
- nonnumeric prices, reviews, discounts, or ratings;
- ratings outside 0–5;
- discounts outside 0%–100%;
- negative review counts;
- current prices greater than old prices; and
- price ranges or other ambiguous values.

Use a small cleaning log in `Data_Dictionary`:

| Issue | Rows affected | Decision | Reason |
|---|---:|---|---|
| Example: negative review values | Enter count | Convert to absolute values | The sign is treated as a scraping artifact; review counts cannot be negative |

### 5.3 Standardize the columns

Rename fields consistently, for example `Current Price`, `Old Price`, and `Rating`. Apply the following rules:

- **Product:** use `TRIM`/Power Query **Trim** and **Clean**; retain meaningful punctuation.
- **Prices:** remove `KSh`, commas, and extra spaces, then convert to a number. Format cleaned values as `KSh #,##0.00`.
- **Price ranges:** do not simply delete the hyphen. Choose a documented rule such as midpoint, lower bound, or exclusion. A midpoint is useful for analysis, while preserving the original text in a separate field keeps the decision auditable.
- **Discount:** remove `%`, convert to a number, divide by 100 if necessary, and format as Percentage.
- **Review:** investigate the negative sign. If it is a formatting/scraping artifact, use the absolute whole number. Leave a genuinely missing review as blank unless the project owner confirms that blank means zero.
- **Rating:** remove ` out of 5`, convert to a decimal, and leave genuinely missing ratings blank.
- **Duplicates:** remove only records that are duplicates across all relevant source fields. Products with the same name but different prices or feedback may be legitimate listings.

Example formulas for ordinary (non-range) values, assuming the source value is in row 2:

```excel
=VALUE(SUBSTITUTE(SUBSTITUTE(B2,"KSh",""),",",""))
=VALUE(SUBSTITUTE(D2,"%",""))/100
=IF(E2="","",ABS(VALUE(E2)))
=IF(F2="","",VALUE(SUBSTITUTE(F2," out of 5","")))
```

If a price range must use its midpoint, Power Query's **Split Column by Delimiter** is safer than a long formula: remove `KSh` and commas, split on ` - ` into minimum and maximum, convert both to numbers, then calculate `(Minimum + Maximum) / 2`.

### 5.4 Handle missing values honestly

- Do not replace a missing rating with the average rating unless you can justify imputation.
- Do not assume a blank review means zero; “not captured” and “no reviews” are different facts.
- Exclude blanks from metrics that require the missing field and display the relevant valid-record count.
- Add a `Data Status` field (for example, `Complete`, `Missing rating`, or `Missing review`) if it helps users understand coverage.

### 5.5 Validate the cleaned table

Add checks before analysis:

```excel
=IF(OR([@Rating]<0,[@Rating]>5),"Check rating","OK")
=IF(OR([@Discount]<0,[@Discount]>1),"Check discount","OK")
=IF([@[Current Price]]>[@[Old Price]],"Check prices","OK")
```

Spot-check at least ten rows against `Raw_Data`, including blanks, duplicates, negative reviews, and a price range. Refresh the query/PivotTables and confirm that numeric fields summarize with **Sum/Average**, not **Count**.

## 6. Step 2 — Enrich the data

Add the following columns to `tblProducts`.

### Discount amount

```excel
=[@[Old Price]]-[@[Current Price]]
```

Optionally compare the advertised discount with the calculated discount:

```excel
=IFERROR(([@[Old Price]]-[@[Current Price]])/[@[Old Price]],"")
```

Flag material differences rather than overwriting the advertised value.

### Rating category

The brief says **Poor** is below 3, **Average** is 3–4, and **Excellent** is above 4.5. That leaves ratings from 4.1 through 4.5 without a category. Obtain instructor clarification if possible. For an exhaustive three-category dashboard, use and disclose this working assumption: **Poor < 3; Average 3–4.5; Excellent > 4.5**.

```excel
=IF([@Rating]="","Missing",IF([@Rating]<3,"Poor",IF([@Rating]<=4.5,"Average","Excellent")))
```

If the original boundaries must be followed literally, label 4.1–4.5 as `Unclassified` instead of forcing those products into another group.

### Discount category

Use **Low Discount < 20%**, **Medium Discount 20%–40%**, and **High Discount > 40%**. This assigns exactly 40% to Medium.

```excel
=IF([@Discount]="","Missing",IF([@Discount]<20%,"Low Discount",IF([@Discount]<=40%,"Medium Discount","High Discount")))
```

### Price category

Define defensible thresholds rather than choosing arbitrary values. A reproducible approach uses the first and third quartiles calculated in cells on `Analysis`:

```excel
=QUARTILE.INC(tblProducts[Current Price],1)
=QUARTILE.INC(tblProducts[Current Price],3)
```

Name those cells `Price_Q1` and `Price_Q3`, then use:

```excel
=IF([@[Current Price]]="","Missing",IF([@[Current Price]]<=Price_Q1,"Low Price",IF([@[Current Price]]<=Price_Q3,"Medium Price","High Price")))
```

Record the final KSh thresholds in `Data_Dictionary` so that dashboard users know what each category means.

### Engagement and performance flags

For “strong customer engagement,” define a measurable rule, such as review count at or above the 75th percentile. You may also create flags for:

- high discount and low rating;
- high discount and low engagement;
- many reviews and average rating; and
- strong engagement and excellent rating.

State the exact thresholds; do not select products subjectively.

## 7. Step 3 — Analyze the data

### 7.1 Descriptive statistics and KPIs

Calculate:

| Metric | Example Excel formula |
|---|---|
| Total products | `=ROWS(tblProducts[Product])` |
| Average current price | `=AVERAGE(tblProducts[Current Price])` |
| Average old price | `=AVERAGE(tblProducts[Old Price])` |
| Average discount | `=AVERAGE(tblProducts[Discount])` |
| Average rating | `=AVERAGE(tblProducts[Rating])` |
| Total reviews | `=SUM(tblProducts[Review])` |
| Most expensive price | `=MAX(tblProducts[Current Price])` |
| Least expensive price | `=MIN(tblProducts[Current Price])` |

Return the product names associated with the extremes:

```excel
=XLOOKUP(MAX(tblProducts[Current Price]),tblProducts[Current Price],tblProducts[Product])
=XLOOKUP(MIN(tblProducts[Current Price]),tblProducts[Current Price],tblProducts[Product])
```

If ties exist, report all tied products using `FILTER` rather than showing only the first match.

### 7.2 Relationship analysis

Create three scatter plots, using one product per point:

1. Discount (x) versus reviews (y);
2. Rating (x) versus reviews (y); and
3. Current price (x) versus rating (y).

Add a linear trendline, display the equation and R-squared value, and calculate Pearson correlations:

```excel
=CORREL(tblProducts[Discount],tblProducts[Review])
=CORREL(tblProducts[Rating],tblProducts[Review])
=CORREL(tblProducts[Current Price],tblProducts[Rating])
```

Because blanks can misalign arrays in some Excel versions, a PivotTable or filtered helper range containing only complete pairs may be required. Interpret direction and magnitude carefully. Correlation does not establish that discounts cause reviews, and review totals may reflect product age, visibility, or sales volume that the dataset does not capture.

### 7.3 Ranked product analysis

Create ranked tables for:

- top 5 and bottom 5 products by rating;
- top 10 products by discount;
- top 10 products by reviews;
- top 10 products by rating;
- products with high discounts but low ratings;
- products with high discounts but low engagement; and
- products with many reviews but average ratings.

Use a PivotTable (**Rows:** Product; **Values:** relevant measure), sort largest-to-smallest or smallest-to-largest, and apply **Value Filters > Top 10**. Use review count as a tie-breaker for rating ranks and rating as a tie-breaker for review ranks. Exclude missing ratings from rating lists and disclose how ties are handled.

### 7.4 Seller-performance questions

Translate the ranked tables into evidence-based answers:

- **Strong demand proxy:** Which products have review counts above the engagement threshold?
- **Pricing/marketing attention:** Which products combine weak engagement or weak ratings with high price/high discount?
- **Many reviews, average experience:** Which products have high engagement but fall in the Average rating category?
- **Promotion inefficiency:** Which high-discount products remain below the engagement threshold?

## 8. Step 4 — Build PivotTables and PivotCharts

Suggested PivotTables:

| PivotTable | Rows | Values | Recommended visual |
|---|---|---|---|
| Rating mix | Rating Category | Count of Product | Doughnut or column chart |
| Discount mix | Discount Category | Count of Product | Column chart |
| Price vs rating | Price Category | Average of Rating | Column chart |
| Engagement by discount | Discount Category | Average/Sum of Review | Column chart |
| Top products by rating | Product | Average of Rating | Horizontal bar chart |
| Top products by reviews | Product | Sum of Review | Horizontal bar chart |
| Top products by discount | Product | Average of Discount | Horizontal bar chart |

Use **Insert > Slicer** for `Rating Category`, `Discount Category`, and `Price Category`. For each slicer, open **Report Connections** (or **PivotTable Connections**) and connect it to every compatible PivotTable. Test each slicer independently and in combination.

## 9. Step 5 — Design the dashboard

Use a single-screen layout that can be read without scrolling at a normal zoom level:

```text
+------------------------------------------------------------------+
| Title, refresh date, and slicers                                 |
+------------------------------------------------------------------+
| Total Products | Avg Price | Avg Discount | Avg Rating | Reviews |
+------------------------------------------------------------------+
| Top 10 by Rating | Top 10 by Reviews | Top 10 by Discount        |
+------------------------------------------------------------------+
| Discount vs Reviews | Rating vs Reviews | Price vs Rating         |
+------------------------------------------------------------------+
| Rating Mix | Discount Mix | Key Insights / Recommendations       |
+------------------------------------------------------------------+
```

Dashboard standards:

- use a consistent color palette, font, spacing, and number format;
- show KSh on prices, `%` on discounts, one decimal on ratings, and thousands separators on reviews;
- sort bar charts logically and prefer horizontal bars for long product names;
- use dynamic chart titles where practical;
- avoid 3-D charts, unnecessary legends, crowded labels, and misleading axes;
- use conditional formatting to highlight weak ratings, large discounts, and strong engagement;
- show `N/A` instead of a misleading zero where no filtered records exist; and
- include a small note explaining category thresholds, missing-value handling, and that reviews are an engagement proxy.

## 10. Business insights and recommendations

Write three to five concise findings after completing the workbook. Each finding should contain:

1. **Evidence:** a metric, comparison, rank, or correlation;
2. **Meaning:** why it matters to Jumia or a seller;
3. **Action:** a specific next step; and
4. **Caveat:** any limitation that changes interpretation.

Use a structure such as:

> Products in the ___ discount category average ___ reviews versus ___ in the ___ category. This suggests ___. Sellers should test ___. However, this dataset does not include sales or listing age, so the pattern should not be treated as causal.

Recommendations might include testing discount bands, improving listing content for visible but average-rated products, investigating quality issues on highly discounted/low-rated items, or promoting products that combine excellent ratings with strong engagement. Only recommend actions supported by your completed analysis.

## 11. Quality-assurance checklist

Before submission, confirm that:

- [ ] `Raw_Data` is unchanged and the cleaned data is in an Excel Table.
- [ ] Every cleaning choice is recorded in the cleaning log.
- [ ] Duplicate removal uses a documented key.
- [ ] Prices, discounts, reviews, and ratings are numeric.
- [ ] Missing values are not silently converted to zero.
- [ ] Price ranges and negative reviews have documented treatment.
- [ ] Discount amount equals old price minus current price.
- [ ] Categories cover every valid record with documented boundaries.
- [ ] KPI values agree with the cleaned table/PivotTables.
- [ ] Rankings exclude missing values and use stated tie-breakers.
- [ ] Scatter plots use the correct x- and y-axes.
- [ ] Slicers control all intended PivotCharts and KPIs.
- [ ] Charts have titles, readable labels, units, and appropriate formats.
- [ ] Insights distinguish correlation from causation.
- [ ] The workbook opens without broken links or formula errors.
- [ ] All PivotTables are refreshed before saving.

## 12. README for your own GitHub repository

Your submitted repository should be easy to navigate. A suggested structure is:

```text
jumia-product-performance-dashboard/
├── README.md
├── data/
│   └── Excel_jumia_dataset.csv
├── dashboard/
│   └── jumia_product_dashboard.xlsx
└── images/
    ├── raw-data.png
    ├── cleaned-data.png
    ├── pivot-tables.png
    └── dashboard.png
```

Your project `README.md` should explain the objective, dataset, tools, cleaning decisions, formulas, analysis, dashboard features, key findings, recommendations, limitations, file structure, and how to open/use the workbook. Do not publish private information or unrelated files.

## 13. Dev.to technical article

Publish an individual article titled:

> **Building an Interactive Excel Dashboard for E-commerce Product Analysis: A Case Study of Jumia Products**

Recommended outline:

1. Project introduction and objective;
2. Dataset and business questions;
3. Initial data-quality audit;
4. Cleaning and preparation decisions;
5. Excel formulas and enrichment fields;
6. PivotTable and analysis workflow;
7. Dashboard design and slicer connections;
8. Key findings;
9. Business recommendations;
10. Limitations and lessons learned; and
11. Links to the GitHub repository and dashboard file.

Include clear screenshots of the raw data, cleaned data, formulas/calculations, PivotTables, charts, final dashboard, and insight summary. Crop screenshots to remove personal information, keep text readable, and add a caption explaining what each image demonstrates. Write the article in your own words and make sure reported figures match the final workbook.

## 14. Final deliverables and submission

Submit:

1. an `.xlsx` workbook containing the original data, cleaned data, calculations, analysis, PivotTables/PivotCharts, slicers, and final dashboard;
2. a documented GitHub repository with a project `README.md`; and
3. the published Dev.to technical article.

Email the following links to **ch9datascience@luxdevhq.com**:

1. GitHub repository link; and
2. Dev.to article link.

The stated deadline is **Saturday, September 5 at 5:00 PM**. Confirm the applicable year and time zone with the instructor, and verify that both links are accessible before sending the email.

## 15. Expected learning outcomes

After completing this project, you should be able to:

- clean and prepare real-world business data in Excel;
- apply formulas for transformation and analysis;
- build and refresh PivotTables and PivotCharts;
- create an interactive dashboard with slicers and conditional formatting;
- assess price, discount, rating, and engagement relationships; and
- communicate defensible findings, limitations, and business recommendations.

