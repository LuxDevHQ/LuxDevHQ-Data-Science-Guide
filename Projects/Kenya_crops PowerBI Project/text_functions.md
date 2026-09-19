# Practice Questions: DAX Text Functions
## Kenya Crops Dataset

**Dataset Fields:**
- Farmer Name
- Crop Type
- County
- Market
- Revenue (KES)
- Yield (Kg)
- Planting Date

---

## Section A: EXACT Function

1. Create a measure that checks if the Crop Type is exactly "Maize" (case-sensitive).
2. Create a measure that verifies whether the Farmer Name is exactly "Mary Wanjiku".
3. Create a column that returns TRUE if County is exactly "Nakuru" and FALSE otherwise.

---

## Section B: FIND Function

1. Create a calculated column that finds the position of the word "Pot" in Crop Type.
2. Write a DAX formula that searches for the word "John" in Farmer Name and returns 0 if not found.
3. Find the position of the substring "Market" inside the Market column.

---

## Section C: FORMAT Function

1. Create a measure that formats total revenue as currency.
2. Format the Planting Date column to display in the format: `March 10, 2023`
3. Create a column that formats Yield (Kg) to two decimal places.

---

## Section D: LEFT Function

1. Create a column that extracts the first 3 letters of Crop Type.
2. Extract the first 4 characters from the County name.
3. Extract the first 5 characters from Farmer Name.

---

## Section E: RIGHT Function

1. Extract the last 3 letters from Crop Type.
2. Extract the last 4 characters from the Market name.
3. Extract the last 6 characters from Farmer Name.

---

## Section F: LEN Function

1. Create a column that calculates the length of the Crop Type name.
2. Find the number of characters in each Farmer Name.
3. Determine the length of the County name.

---

## Section G: LOWER Function

1. Convert Crop Type into lowercase.
2. Convert Farmer Name into lowercase.
3. Convert County names into lowercase.

---

## Section H: UPPER Function

1. Convert Crop Type into uppercase.
2. Convert Farmer Name into uppercase.
3. Convert Market names into uppercase.

---

## Section I: TRIM Function

1. Create a column that removes extra spaces in Crop Type.
2. Remove leading or trailing spaces in Farmer Name.
3. Clean the County column by removing extra spaces.

---

## Section J: CONCATENATE Function

1. Combine Crop Type and Farmer Name into one column.
2. Combine County and Market into one text field.
3. Create a column that combines Farmer Name + Crop Type.

   **Example Output:** `Mary Wanjiku - Potatoes`

---

## Bonus Practice (Real Analysis)

1. Create a column showing First 3 letters of Crop Type + County.

   **Example:** `MAI - Nakuru`

2. Create a column combining Farmer Name + Planting Date.

   **Example:** `John Kamau planted on March 10, 2023`

3. Convert all Farmer Names to uppercase and remove extra spaces.
```
