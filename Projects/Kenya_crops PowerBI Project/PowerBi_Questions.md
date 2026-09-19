# Practice Questions – Power BI DAX (Crops Dataset)

## 1. Math & Statistical Functions

1. Write a DAX measure to calculate the **Total Revenue** from the `Revenue (KES)` column.

2. Using `SUMX()`, calculate the **Total Fertilizer Cost** by multiplying `Yield (Kg)` with `Fertilizer Cost (KES/Kg)`.

3. Create a measure that finds the **Average Market Price** of crops.

4. Use `AVERAGEX()` to calculate the **Average Profit per crop** = (Yield × Price – Cost of Production).

5. Write a measure to return the **Median Yield** of all crops.

## 2. Filter Functions

1. Use `FILTER()` inside `CALCULATE()` to get the **Total Revenue** for crops with Revenue > 5,000 KES.

2. Write a DAX measure with `CALCULATE()` to find **Total Revenue** for crops grown in the "West" region.

3. Create a measure with `HASONEVALUE()` to check if only one Crop Type is selected in a slicer.

4. Using `ALL()`, calculate the overall revenue ignoring all filters.

5. Use `ALLEXCEPT()` to calculate Revenue by Region, ignoring all other filters.

6. With `REMOVEFILTERS()`, calculate Total Revenue ignoring Region filter only.

## 3. Logical Functions

1. Write a DAX formula with `IF()` to return "High" if Profit Margin > 0.2, else "Low".

2. Using `AND()`, calculate revenue only when Revenue > 5000 and Profit Margin > 0.2.

3. With `OR()`, return TRUE if either Revenue > 5000 or Profit Margin > 0.2.

4. Write a nested `IF()` that classifies Revenue into "High", "Medium", or "Low".

5. Use `SWITCH()` to return "Western Region", "Eastern Region", "Central Region", or "Unknown Region" based on Region.

6. Write a measure using `IFERROR()` that divides Revenue by Quantity but returns 0 if there's an error.

## 4. Text Functions

1. Use `EXACT()` to check if a Crop Type is exactly "Potatoes".

2. Use `FIND()` to find the position of "John" in the Farmer Name column.

3. Format the Total Revenue as currency using `FORMAT()`.

4. Extract the first 3 characters of Crop Type with `LEFT()`.

5. Extract the last 5 characters of Farmer Name with `RIGHT()`.

6. Use `LEN()` to find the length of each Farmer Name.

7. Convert Crop Type to lowercase with `LOWER()` and uppercase with `UPPER()`.

8. Use `TRIM()` to remove spaces in Crop Type.

9. Create a column that concatenates Crop Type and Farmer Name with a space in between.

## 5. Date & Time Intelligence

1. Create a Date Table for 2023 using `CALENDAR()`.

2. Use `DATEDIFF()` to calculate the number of days between Planting Date and `Today()`.

3. Extract the Month number and Year from the Planting Date using `MONTH()` and `YEAR()`.

4. Write a measure to calculate **YTD Revenue** using `TOTALYTD()`.

5. Using `SAMEPERIODLASTYEAR()`, calculate the Revenue Last Year for comparison.

6. Write a measure with `DATEADD()` to calculate Revenue 3 Months Ago.

7. Use `DATESBETWEEN()` to calculate Revenue only between July 1, 2022 and Sept 30, 2022.
