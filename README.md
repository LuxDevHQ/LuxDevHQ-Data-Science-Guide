## **LuxDevHQ Data Science and Analytics Curriculum.**

Please note that this program is scheduled to run for 6 months. However, depending on the students’ pace, it may extend for up to 9 months, Mode of Delivery, Physical and online live instructor-led classes with hands-on projects.

**For any inquiries, please reach out via call or WhatsApp: 0798166628 / 0796448232.**

> Regardless of how long the program takes, the fees remain unchanged. The fees will not be affected. At some point, students may request an additional week or two for projects or for certain concepts to be explained more thoroughly.

- Note that this guide is  structured as a progressive pathway: spreadsheet analytics → BI → SQL → Python → statistics → machine learning → deep learning → NLP/RAG → deployment → capstone delivery.

### **Table of Contents.**

1. [Purpose of This Guide](#purpose-of-this-guide)
2. [Who This Guide Is For](#who-this-guide-is-for)
3. [How to Use the Guide](#how-to-use-the-guide)
4. [Curated Learning Path](#curated-learning-path)
5. [Integrated Skill Map](#integrated-skill-map)
6. [Recommended Assessment Strategy](#recommended-assessment-strategy)
7. [Capstone Portfolio Sequence](#capstone-portfolio-sequence)
8. [Dataset Inventory](#dataset-inventory)
9. [Facilitator Notes](#facilitator-notes)
10. [Merged Source Content](#merged-source-content)
    - [Core Curriculum](#core-curriculum)
    - [Power BI DAX Practice Bank](#power-bi-dax-practice-bank)
    - [DAX Text Functions Drill](#dax-text-functions-drill)
    - [RAG In-Class Activity](#rag-in-class-activity)
    - [Adaptive Learning Capstone](#adaptive-learning-capstone)
    - [Japan Car Import Advisory Capstone](#japan-car-import-advisory-capstone)
11. [Appendix: Curation Checklist](#appendix-curation-checklist)

---

### Purpose of This Guide

This guide consolidates the repository's curriculum, practice activities, datasets, and capstone briefs into one coherent learning document. 

The goal is to help learners move from tool literacy to practical data science execution:

- **Analyze data** with Excel, Power BI, SQL, and Python.

- **Explain data** with statistics, visualization, dashboards, and business storytelling.
  
- **Model data** with supervised learning, unsupervised learning, deep learning, and evaluation methods.
  
- **Build AI products** with retrieval-augmented generation, APIs, databases, and deployment workflows.
  
- **Graduate with a portfolio** of dashboards, notebooks, model pipelines, RAG systems, and capstone products.

### **Who This Guide Is For:**

- This program is designed for beginners who need a structured path into data analytics, data science, and AI engineering; instructors who need a single facilitation document for a 6-month to 9-month program; and learners who prefer project-based practice instead of theory only.


### **How to Use the Guide.**

1. **Start with the roadmap.** Use the curated learning path to understand the recommended order.
2. **Teach in layers.** Introduce concepts, demonstrate them, then assign practice questions or mini-projects.
3. **Use the Kenya Crops dataset repeatedly.** It supports Excel cleaning, Power BI DAX, SQL-style thinking, and Python EDA.
4. **Build portfolio artifacts every month.** Do not wait until the final capstone to create deliverables.
5. **Use capstones as integration points.** The RAG activity, adaptive learning system, and car import advisory project should connect analytics, modeling, APIs, and communication.

### **Curated Learning Path.**

| Phase | Recommended Timing | Focus | Primary Deliverables |
|---|---:|---|---|
| 1 | Month 1 | Excel foundations, formulas, cleaning, PivotTables, dashboards | Cleaned workbook + pivot dashboard |
| 2 | Month 1 | Power BI, Power Query, DAX, modeling, dashboards | Interactive Power BI report using Kenya Crops data |
| 3 | Month 2 | SQL querying, joins, aggregations, CTEs, window functions | SQL analysis notebook or query pack |
| 4 | Month 2 | Python, pandas, EDA, automation | Python EDA notebook + reusable cleaning script |
| 5 | Pre-ML bridge | Statistics, distributions, probability, hypothesis testing | Statistics interpretation memo |
| 6 | Month 3 | Supervised ML: regression, classification, trees, ensembles | Baseline ML model with evaluation report |
| 7 | Month 4 | Unsupervised learning, evaluation, feature engineering, pipelines | Reproducible sklearn pipeline |
| 8 | Months 5 | Deep learning foundations, ANN/CNN, transfer learning | Keras model notebook + model card |
| 9 | Month 6 | NLP, embeddings, semantic search, RAG | ChromaDB RAG prototype |
| 10 | Month 7 | Deployment, APIs, databases, final capstone | Deployed API or app + final presentation |

## Integrated Skill Map

| Skill Area | Learner Should Be Able To | Evidence |
|---|---|---|
| Spreadsheet analytics | Clean, transform, summarize, and visualize tabular data | Excel workbook and dashboard |
| BI and dashboarding | Build data models, measures, slicers, and executive dashboards | Power BI report and DAX measures |
| SQL analytics | Query relational data and communicate findings | Query file with joins, CTEs, and windows |
| Python analytics | Load, clean, visualize, and automate analysis | Notebook and Python scripts |
| Statistics | Choose metrics, interpret distributions, compare groups, avoid misleading claims | Written analysis memo |
| Machine learning | Train, validate, tune, and explain models | Model pipeline and evaluation report |
| Deep learning | Build neural networks for structured, image, or text data | Keras/TensorFlow notebook |
| NLP and RAG | Chunk documents, create embeddings, retrieve context, and evaluate retrieval quality | RAG prototype with retrieval logs |
| Product engineering | Connect AI/ML to APIs, databases, and user workflows | FastAPI service or deployed app |
| Storytelling | Convert analysis into decisions and next steps | Slide deck and demo video |

## Recommended Assessment Strategy

| Assessment Type | Weight | What to Evaluate |
|---|---:|---|
| Weekly labs | 20% | Correct use of tools, reproducibility, and data-cleaning discipline |
| Quizzes and practice banks | 15% | Concept retention, formulas, DAX, SQL, and ML vocabulary |
| Mini-projects | 25% | End-to-end analysis, visual clarity, model reasoning, and communication |
| Peer review | 10% | Code readability, dashboard usability, and explanation quality |
| Final capstone | 30% | Business framing, technical implementation, evaluation, deployment readiness, and presentation |

---

### **Core Curriculum.**
 
### **Microsoft Excel for Data Analysis**
 
#### **Excel Basics & Navigation**
- What is Excel and why it's important
- Excel interface: ribbons, sheets, cells, rows, columns
- Data types: text, numbers, dates
- Entering and editing data
- Basic formatting: font, color, alignment, number formatting
- Saving and organizing files
- Keyboard shortcuts
 
#### **Working with Formulas and Functions**
- Cell references: relative, absolute, mixed
- Arithmetic operations: +, -, *, /
- Basic functions:
  - SUM(), AVERAGE(), MIN(), MAX(), COUNT(), COUNTA()
- Logical functions:
  - IF(), AND(), OR(), NOT()
- Text functions:
  - CONCATENATE(), LEFT(), RIGHT(), LEN(), TRIM(), UPPER(), LOWER(), PROPER()
- Date & Time functions:
  - TODAY(), NOW(), DATEDIF(), YEAR(), MONTH(), DAY()
 
#### **Data Cleaning & Sorting/Filtering**
- Remove duplicates
- Find & Replace
- Text to Columns
- Error checking: ISERROR(), IFERROR()
- Sorting data (A-Z, Z-A, custom sort)
- Filtering data (AutoFilter, Custom Filter)
- Freeze panes, hide/unhide rows/columns
 
#### **Data Analysis with Pivot Tables, Charts, and Dashboards**
- Introduction to PivotTables
  - Creating PivotTables
  - Rows, Columns, Values, Filters
  - Summarize by count, average, percentage
- PivotCharts
  - Column, Bar, Line, Pie Charts
- Slicers for interactive filtering
- Dashboard creation
  - Combining PivotTables, Charts, and Slicers
  - Best practices for layout and design
 
---
 
### Power BI for Data Visualization
---
 
### **Introduction to Power BI and Power Query Editor (Data Transformation)**
 
#### Overview & Setup
- What is Power BI? (Comparison with Excel & Tableau)
- Power BI Components: Desktop, Service, Mobile
- Installing Power BI Desktop
- Power BI Desktop Interface Overview
 
#### Getting Data
- Importing Data from Excel, CSV, Web
- Understanding Data Types and Field Formatting
 
#### Power Query Editor
- Opening Power Query Editor
- Removing Rows, Columns, and Duplicates
- Changing Data Types, Renaming Columns
- Splitting & Merging Columns
- Using “Replace Values”
- Applied Steps, Reordering, and Removing Steps
 
#### Combining Queries
- **Merge Queries** (SQL-style joins)
- **Append Queries** (Union of datasets)
 
---
 
### **DAX Basics – Measures, Calculated Columns & Aggregations**
 
#### Introduction to DAX
- What is DAX (Data Analysis Expressions)?
- Syntax Rules: =, (), [], Table & Column references
 
#### Calculated Columns vs Measures
- When to use each
- Creating new fields using Calculated Columns
- Building Measures for aggregations
 
#### Common DAX Functions
- **Aggregation**: SUM(), AVERAGE(), COUNT(), COUNTROWS(), DISTINCTCOUNT()
- **Logical**: IF(), SWITCH(), AND(), OR()
- **Date**: TODAY(), NOW(), YEAR(), MONTH(), DATEDIFF()
- **Text**: CONCATENATE(), LEFT(), RIGHT(), LEN()
 
#### Practical Use Cases
- Total Revenue, Profit Margin, % Growth
- IF(Sales > 50000, “High”, “Low”)
- Number of Orders per Customer
 
---
 
### **Data Modeling, Relationships & Joins**
 
#### Data Modeling Concepts
- What is a Data Model?
- Star Schema vs Snowflake Schema
- Importance of Fact and Dimension tables
 
#### Relationships in Power BI
- One-to-Many and Many-to-One relationships
- Creating & Managing Relationships in Model View
- Active vs Inactive Relationships
 
#### Data Joins
- Relationship-based vs Merge Query joins
- Cardinality (One-to-One, One-to-Many)
- Cross filter direction
 
#### Modeling Best Practices
- Hiding unnecessary columns
- Using Lookup Tables
- Creating Role-Playing Dimensions (e.g. Order Date vs Delivery Date)
 
---
 
### **Visualizations, Charts & Dashboards**
 
#### Basic Visuals
- Bar Chart, Column Chart, Line Chart
- Pie & Donut Charts
- Card, KPI
- Table & Matrix
 
#### Advanced Visuals
- Tree Map, Funnel, Gauge
- Maps: Filled Map, Shape Map, ArcGIS Map
- Custom Visuals (via AppSource)
 
#### Interactivity
- Visual-Level, Page-Level, Report-Level Filters
- Slicers (Text, Date, Dropdowns)
- Drill-down & Drill-through
- Tooltips, Bookmarks, Buttons
 
#### Dashboard Building
- Designing a complete report page
- Adding Titles, Backgrounds, Logos, Images
- Aligning and Formatting Visuals
- Creating Navigation Buttons
 
#### Publishing & Sharing
- Publishing to Power BI Service
- Sharing Reports & Dashboards
- Setting Scheduled Refresh
 
---
 
 
### SQL for Data Analysis
 
#### **Introduction to SQL, Table Creation and Manipulation and SQL Keywords**
- What is SQL, relational databases, and DBMS
- CREATE, DROP, ALTER TABLE
- INSERT, UPDATE, DELETE data
- SELECT and FROM
- ORDER BY ASC/DESC
- LIMIT and OFFSET
- GROUP BY and HAVING
 
#### **Aggregations and Operators**
- Comparison Operators : =, <>, >, <, BETWEEN, IN, LIKE
- Aggregates: COUNT(), SUM(), AVG(), MIN(), MAX()
- String Functions: CONCAT(), LENGTH(), SUBSTRING(), UPPER(), LOWER(), REPLACE()
- Date Functions: NOW(), CURRENT_DATE, EXTRACT(), AGE()
- Math Functions: ROUND(), CEIL(), FLOOR(), MOD(), POWER(), ABS()
- Window functions:
  - ROW_NUMBER(), RANK(), DENSE_RANK()
  - SUM(), AVG() OVER(PARTITION BY...)
  - LEAD(), LAG()
- CASE WHEN THEN ELSE END
 
#### **SQL Joins and Relationships**
- Primary and foreign keys
- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- FULL OUTER JOIN
- CROSS JOIN
- Self Join and aliasing
- Combining joins with aggregations and filters
- Practice: build reports using joins from multiple tables
 
 
#### **Advanced SQL (CTEs, Subqueries, Stored Procedures)**
- Subqueries: scalar, correlated, IN/EXISTS
- Common Table Expressions (CTEs)
- Stored Procedures and parameters (Intro)
 
 
---
 
### Python for Data Analysis
 
#### **Python Basics**
- What is Python and its use in data
- Setting up Python (Jupyter, Colab, VSCode)
- Variables and data types (int, float, string, bool)
- Lists, Tuples, Dictionaries, Sets
- Operators and expressions
- Conditional statements: if, elif, else
- Loops: for, while
- Functions and basic I/O
 
#### **Working with Data using Pandas**
- Importing data: CSV, Excel
- DataFrames and Series
- Inspecting data: head(), tail(), info(), describe()
- Indexing and selection: loc[], iloc[], slicing
- Filtering data
- Sorting, renaming columns
- Adding, deleting columns
 
#### **Data Cleaning and Analysis**
- Handling missing data: isnull(), fillna(), dropna()
- Duplicates: duplicated(), drop_duplicates()
- Changing data types
- GroupBy and Aggregations
- Merging and joining DataFrames
- Pivot tables with pandas
- Basic visualizations: matplotlib, seaborn
 
#### **Projects and Automation with Python**
- Real-world mini project (sales data, HR data, etc.)
- Creating summary reports
- Exporting cleaned data to Excel/CSV
- Automating tasks with loops and functions
- Intro to working with APIs
- Final recap and practice challenges
 
---


### Statistics Foundations (Pre-ML)

---

#### **Descriptive Statistics & Data Distributions**
- Types of data: numerical (continuous, discrete) and categorical (nominal, ordinal)
- Measures of central tendency: mean, median, mode — and when to use each
- Measures of spread: range, variance, standard deviation (sd), IQR
- Understanding outliers and their effect on statistics
- What is a distribution? Types of distribution overview
-  Normal distribution (bell curve) and the 68-95-99.7 rule.  
- Skewness: right-skewed vs left-skewed distributions
- Visualising distributions in Python: histograms and box plots

---

#### **Probability, Correlation & Hypothesis Testing**
- Basic probability: P(event) = favourable outcomes / total outcomes
- Conditional probability: P(A | B) and real-world examples
- Pearson correlation coefficient: strength and direction (−1 to +1)
- Correlation vs causation — the ice cream and drowning example
- Creating and reading a correlation heatmap in Python
- Hypothesis testing: H0 (null) vs H1 (alternative hypothesis)
- The p-value: what it means and the alpha = 0.05 threshold
- Type-I and Type-II errors
- Types of Chi-Square tests and how to conduct a chi-square test
- Parametric tests:
  - T-tests: independent samples t-test, paired samples t-test, one-sample t-test
  - Z-tests
  - One-Way ANOVA
- Non-parametric tests:
  - Mann-Whitney U test
  - Wilcoxon Signed-Rank test
  - Kruskal-Wallis test
- Choosing between parametric and non-parametric tests: assumptions (normality, sample size, data type)
- Running these tests in Python using `scipy.stats`
- Connecting statistics to ML: where each concept shows up in algorithms

---


## Machine Learning & Deep Learning Course Plan

---

### Introduction to Machine Learning & Regression

---

#### [**What is Machine Learning?**](#introduction-to-ml-basic-concepts)
**Topic:** Introduction to ML & Types of Learning
- Introduction to ML and real-world applications
- Types of ML: Supervised vs. Unsupervised vs. Reinforcement
- Concept of datasets: features, labels, training vs. testing data

---

#### [**Supervised Learning Flow & Use Cases**](#introduction-to-ml-basic-concepts)
**Topic:** Supervised Learning in Practice
- Supervised learning definition and end-to-end workflow
- Use cases of classification and regression
- Hands-on: exploring a real dataset (e.g., Iris or Titanic) and identifying features and labels

---

#### [**Linear Regression**](#simple-linear-regression--beginner-notes)
**Topic:** Simple & Multiple Linear Regression
- Linear regression: line of best fit, cost function, gradient descent intuition
- Evaluation metrics: MSE, RMSE, R2 score
- Hands-on: simple linear regression example (e.g., study hours vs exam score)

---

#### [**Multiple Linear Regression & Regularization**](#lasso--ridge-regression--beginner-notes)
**Topic:** Multiple Linear Regression, Lasso & Ridge
- Multiple linear regression: multiple features and feature interpretation
- Lasso regression: L1 penalty and feature sparsity
- Ridge regression: L2 penalty and coefficient shrinkage
- Hands-on: predicting house prices with Linear, Lasso, and Ridge

---

### Classification Algorithms

---

#### [**Logistic Regression**](#logistic-regression--beginner-friendly-notes-with-analogies--python-example)
**Topic:** Binary Classification & Logistic Regression
- Understanding classification problems vs regression problems
- Logistic regression intuition and the sigmoid function
- Decision boundaries and how the model draws the line
- Hands-on: binary classification with logistic regression

---

#### [**K-Nearest Neighbors (KNN)**](#k-nearest-neighbors-knn--beginner-friendly-notes-with-analogies--python-example)
**Topic:** KNN & Classification Evaluation Metrics
- How KNN works: distance, voting, and the role of K
- Choosing the right K: underfitting vs overfitting
- Evaluation techniques: accuracy, confusion matrix, precision, recall, F1 score
- Hands-on: KNN classification with sklearn

---

#### [**Decision Trees – Fundamentals**](#decision-trees--beginner-friendly-notes-with-analogies--python-example)
**Topic:** Decision Tree Structure & Splitting
- How decision trees split data: Gini impurity vs entropy
- Growing a tree: recursive splits and stopping criteria
- Overfitting in trees: why deep trees memorise instead of learn
- Hands-on: building and visualising a decision tree

---

#### **Decision Trees – Tuning & Evaluation**
**Topic:** Controlling Decision Tree Complexity
- Controlling overfitting: `max_depth`, `min_samples_split`, `min_samples_leaf`
- Feature importance: which features does the tree rely on most?
- Hands-on: comparing decision trees at different depths on a real dataset

---

### Ensemble Methods

---

#### [**Random Forest – Introduction**](#ensemble-learning-techniques--beginner-friendly-notes)
**Topic:** Ensemble Learning & Bagging
- Why one model isn't always enough: the wisdom of the crowd
- Bagging (Bootstrap Aggregating) explained with analogy
- Random Forest: how it builds and aggregates decision trees

---

#### [**Random Forest – Advanced Topics**](#deep-dive-into-ensemble-methods)
**Topic:** Random Forest in Practice
- Feature importance in Random Forest
- Bias-variance tradeoff: diagnosing underfitting and overfitting
- Hands-on: Random Forest classifier/regressor on a real dataset

---

#### **Gradient Boosting**
**Topic:** Boosting Concept & Gradient Boosting Algorithm
- Boosting vs bagging: the key conceptual difference
- Gradient Boosting intuition: correcting errors sequentially
- Key hyperparameters: `n_estimators`, `learning_rate`, `max_depth`

---

#### **XGBoost & Ensemble Comparison**
**Topic:** XGBoost & Choosing the Right Ensemble
- XGBoost: how it builds on and improves Gradient Boosting
- Hands-on: train and compare XGBoost vs Random Forest vs Decision Tree
- When to choose which ensemble method

---

### Unsupervised Learning & Dimensionality Reduction

---

#### [**Introduction to Unsupervised Learning & K-Means**](#clustering-and-k-means-in-unsupervised-learning)
**Topic:** Clustering & K-Means Algorithm
- What is unsupervised learning? Real-world applications
- K-Means: centroids, the algorithm step-by-step, choosing K
- Inertia and the Elbow Method for finding the optimal number of clusters

---

#### [**Hierarchical Clustering**](#hierarchical-clustering-in-unsupervised-learning)
**Topic:** Hierarchical Clustering & Customer Segmentation
- Understanding and reading dendrograms
- Agglomerative vs Divisive clustering approaches
- Hands-on project: customer segmentation using hierarchical clustering

---

#### [**Dimensionality Reduction – PCA**](#dimensionality-reduction)
**Topic:** Principal Component Analysis
- Curse of dimensionality: why too many features cause problems
- PCA: reducing dimensions while preserving variance
- Explained variance ratio: how many components to keep
- Use cases: speeding up training and removing noise

---

#### **t-SNE & Visualisation**
**Topic:** t-SNE for High-Dimensional Visualisation
- t-SNE: how it maps high-dimensional data to 2D/3D
- PCA vs t-SNE: interpreting and choosing between them
- Hands-on: visualising clusters with PCA and t-SNE on the digits dataset

---

### Model Evaluation & Feature Engineering

---

#### [**Model Evaluation & Cross-Validation**](#model-evaluation--validation-techniques)
**Topic:** Cross-Validation & Core Evaluation Metrics
- Why a single train/test split isn't reliable enough
- K-fold cross-validation: how it works and why it matters
- Evaluation metrics recap: classification (accuracy, F1) and regression (MSE, R2)

---

#### **Advanced Evaluation – ROC, AUC & Curves**
**Topic:** ROC Curves, AUC & Diagnosing Fit Issues
- ROC curves and AUC score: interpreting the trade-off
- Precision-Recall curves and when to prefer them over ROC
- Diagnosing and fixing overfitting and underfitting

---

#### [**Feature Engineering & Data Preprocessing**](#feature-engineering--data-preprocessing)
**Topic:** Cleaning, Encoding & Scaling
- Handling missing data: deletion vs imputation strategies
- Encoding categorical variables: Label Encoding vs One-Hot Encoding
- Feature scaling: Normalisation (Min-Max) vs Standardisation (Z-score)

---

#### **Feature Selection Techniques**
**Topic:** Selecting the Most Useful Features
- Correlation-based feature selection: dropping redundant features
- Chi-square test for categorical feature relevance
- Recursive Feature Elimination (RFE) with sklearn
- Hands-on: selecting the best features and measuring the impact on model performance

---

### Hyperparameter Tuning, Pipelines & Capstone

---

#### [**Hyperparameter Tuning – Grid & Random Search**](#hyperparameter-tuning--introduction)
**Topic:** Grid Search & Randomised Search
- Difference between model parameters and hyperparameters
- Grid Search: exhaustive search over a defined parameter grid
- Randomised Search: efficient sampling from a parameter space
- Hands-on: tuning a Random Forest with `GridSearchCV` and `RandomizedSearchCV`

---

#### **Advanced Tuning & Bayesian Optimisation**
**Topic:** Bayesian Optimisation with Optuna
- Why random/grid search can be inefficient on large search spaces
- Bayesian optimisation: using past results to pick the next trial
- Hands-on: tuning XGBoost with `Optuna` or `Hyperopt`
- For deeper insights, see: [Deep Dive into Hyperparameter Tuning](#hyperparameter-tuning--deep-dive)

---

#### [**Model Pipelines & Deployment Basics**](#model-pipelines-and-deployment-basics)
**Topic:** Building End-to-End Pipelines & Model Saving
- Using `Pipeline` and `ColumnTransformer` from sklearn
- Combining preprocessing and model steps into a single reusable object
- Saving and loading models with `joblib` or `pickle`
- Introduction to deployment: Streamlit / Flask APIs overview

---
#### [**Real-World Deployment with Flask**](#real-world-deployment-with-flask)
**Topic:** Serving a Trained ML Model as a REST API
- What is an API? The client-server model explained with analogy
- Setting up a Flask application from scratch
- Creating a `/predict` endpoint that accepts JSON and returns a prediction
- Loading a saved `joblib` model inside Flask
- Sending test requests with Postman and `curl`
- Handling errors and edge cases in production
- Deploying to the cloud: pushing to Render (free tier)
  
#### ** Capstone Project Work Session & Presentations** 
**Topic:** Full ML Pipeline – Project & Recap
- Students select a dataset and frame a problem
- Apply the full ML pipeline: EDA → preprocessing → modelling → evaluation → tuning
- Each student/team presents findings, approach, and results
- Q&A, peer feedback, and recap of everything covered in Month 2
- Tips for further learning: books, Kaggle, courses, real-world practice


## Foundations of Deep Learning & ANN Basics

---

#### [Introduction to Deep Learning](#Introduction-to-Deep-Learning)
**Topic:** What is Deep Learning? Understanding Neural Networks  
**Summary:**
- Difference between machine learning and deep learning
- Structure of a biological vs. artificial neuron
- Anatomy of a neural network (layers, nodes, weights, biases)
- Activation functions (ReLU, Leaky ReLU, sigmoid, softmax, tanh)
- Forward pass intuition

---

#### [Building a Neural Network from Scratch](#Building-a-Neural-Network-from-Scratch)
**Topic:** Feedforward Neural Networks  
**Summary:**
- Forward propagation in multi-layer networks
- Loss functions (MSE, Cross-Entropy)
- Backpropagation and gradient descent (basic idea, not full math)
- Training loop overview
- Implementing a simple NN

---

#### [Deep Learning with TensorFlow/Keras](#Deep-Learning-with-TensorFlow/Keras)
**Topic:** Implementing Neural Networks with Keras  
**Summary:**
- Introduction to TensorFlow and Keras
- Building a neural network model using Keras
- Compiling and fitting a model
- Understanding epochs, batch size, learning rate
- Evaluating models and making predictions

---

#### [Improving Model Performance](#Improving-Model-Performance)
**Topic:** Regularization and Optimization  
**Summary:**
- Overfitting and underfitting in deep networks
- Regularization techniques: Dropout, L1/L2 penalties
- Optimizers: SGD, Adam, RMSprop
- Hands-on: tuning models using callbacks, early stopping, etc.

---

### Deep Diving into ANNs

---

#### [Model Design – MLPs, Sequential vs Functional API](#Model-Design-MLPs,-Sequential-vs-Functional-API)
**Topic:** Multi-layer Perceptrons (MLPs) and Keras APIs  
**Summary:**
- MLP architecture and the role of depth in networks
- Using Sequential API for simple stack-like models
- Using Functional API for complex models (multiple inputs/outputs, shared layers, skip connections)
- Hands-on: Building models with both APIs for the same task
- Comparison of readability, flexibility, and real-world use cases

---

#### [Handling Different Data Types](#Handling-Different-Data-Types)
**Topic:** Image, Tabular, and Text Data with ANNs  
**Summary:**
- Preprocessing images for ANN (flattening, normalization)
- Handling categorical and numerical tabular data
- Introduction to tokenizing and embedding text
- Practical: building ANN for tabular and basic image data

---

#### [Model Evaluation and Visualization](#Model-Evaluation-and-Visualization)
**Topic:** Evaluating Deep Learning Models  
**Summary:**
- Accuracy, confusion matrix, precision, recall, F1 score
- ROC curves and AUC
- Visualizing training history: loss vs. accuracy curves
- Model interpretability basics (Grad-CAM preview, attention maps)

---

#### Capstone 1 – ANN Mini Project
**Topic:** Project using ANN on real dataset  
**Summary:**
- Students use datasets (e.g., MNIST, tabular UCI data, sentiment classification)
- Build, tune, and evaluate ANN models
- Start-to-finish pipeline implementation
- Prepare short presentations

---

## CNNs for Image Data

---

#### [Introduction to CNNs](#Introduction-to-CNNs)
**Topic:** Convolutional Neural Networks Basics  
**Summary:**
- Limitations of ANNs with image data
- Convolutions, filters/kernels, stride, padding
- Max pooling and feature maps
- Architecture overview of CNNs (Conv → Pool → FC)
- CNN vs ANN in performance and structure

---

#### [Building CNNs in Keras](#Building-CNNs-in-Keras)
**Topic:** Hands-on with CNN for Image Classification  
**Summary:**
- Creating CNN layers with Conv2D, MaxPooling2D, Flatten, Dense
- Training a CNN on MNIST or CIFAR-10
- Data augmentation with ImageDataGenerator
- Improving accuracy using dropout and regularization

---

#### [Transfer Learning](#Transfer-Learning)
**Topic:** Pre-trained Models (VGG, ResNet, MobileNet)  
**Summary:**
- What is transfer learning and why it's useful  
- Feature extraction vs. fine-tuning  
- Loading pre-trained models with Keras  
- Customizing top layers for new tasks  
- Hands-on: image classification with MobileNetV2 or VGG16  

---

#### Capstone 2 – CNN Project + Course Wrap-up
**Topic:** Final Project + Deep Learning Recap  
**Summary:**
- Students implement an image classifier using CNN or transfer learning
- Full pipeline: preprocessing, training, evaluation
- Project presentations and peer feedback
- Recap of ANN vs CNN, deployment tips, future learning paths (e.g., RNNs, Transformers)

#### Further Study:
- Recurrent Neural Networks(RNN)
- Reinforcement Learning




## Power BI DAX Practice Bank

**Source file:** `PowerBi_Questions.md`  
**Role in guide:** Applied DAX practice questions for mathematical, filter, logical, text, and time-intelligence functions using the Kenya Crops dataset.


## Practice Questions – Power BI DAX (Crops Dataset)

### 1. Math & Statistical Functions

1. Write a DAX measure to calculate the **Total Revenue** from the `Revenue (KES)` column.

2. Using `SUMX()`, calculate the **Total Fertilizer Cost** by multiplying `Yield (Kg)` with `Fertilizer Cost (KES/Kg)`.

3. Create a measure that finds the **Average Market Price** of crops.

4. Use `AVERAGEX()` to calculate the **Average Profit per crop** = (Yield × Price – Cost of Production).

5. Write a measure to return the **Median Yield** of all crops.

### 2. Filter Functions

1. Use `FILTER()` inside `CALCULATE()` to get the **Total Revenue** for crops with Revenue > 5,000 KES.

2. Write a DAX measure with `CALCULATE()` to find **Total Revenue** for crops grown in the "West" region.

3. Create a measure with `HASONEVALUE()` to check if only one Crop Type is selected in a slicer.

4. Using `ALL()`, calculate the overall revenue ignoring all filters.

5. Use `ALLEXCEPT()` to calculate Revenue by Region, ignoring all other filters.

6. With `REMOVEFILTERS()`, calculate Total Revenue ignoring Region filter only.

### 3. Logical Functions

1. Write a DAX formula with `IF()` to return "High" if Profit Margin > 0.2, else "Low".

2. Using `AND()`, calculate revenue only when Revenue > 5000 and Profit Margin > 0.2.

3. With `OR()`, return TRUE if either Revenue > 5000 or Profit Margin > 0.2.

4. Write a nested `IF()` that classifies Revenue into "High", "Medium", or "Low".

5. Use `SWITCH()` to return "Western Region", "Eastern Region", "Central Region", or "Unknown Region" based on Region.

6. Write a measure using `IFERROR()` that divides Revenue by Quantity but returns 0 if there's an error.

### 4. Text Functions

1. Use `EXACT()` to check if a Crop Type is exactly "Potatoes".

2. Use `FIND()` to find the position of "John" in the Farmer Name column.

3. Format the Total Revenue as currency using `FORMAT()`.

4. Extract the first 3 characters of Crop Type with `LEFT()`.

5. Extract the last 5 characters of Farmer Name with `RIGHT()`.

6. Use `LEN()` to find the length of each Farmer Name.

7. Convert Crop Type to lowercase with `LOWER()` and uppercase with `UPPER()`.

8. Use `TRIM()` to remove spaces in Crop Type.

9. Create a column that concatenates Crop Type and Farmer Name with a space in between.

### 5. Date & Time Intelligence

1. Create a Date Table for 2023 using `CALENDAR()`.

2. Use `DATEDIFF()` to calculate the number of days between Planting Date and `Today()`.

3. Extract the Month number and Year from the Planting Date using `MONTH()` and `YEAR()`.

4. Write a measure to calculate **YTD Revenue** using `TOTALYTD()`.

5. Using `SAMEPERIODLASTYEAR()`, calculate the Revenue Last Year for comparison.

6. Write a measure with `DATEADD()` to calculate Revenue 3 Months Ago.

7. Use `DATESBETWEEN()` to calculate Revenue only between July 1, 2022 and Sept 30, 2022.


---


## DAX Text Functions Drill

**Source file:** `text_functions.md`  
**Role in guide:** Focused text-function practice for EXACT, FIND, FORMAT, LEFT, RIGHT, LEN, LOWER, UPPER, TRIM, and CONCATENATE.


## Practice Questions: DAX Text Functions
### Kenya Crops Dataset

**Dataset Fields:**
- Farmer Name
- Crop Type
- County
- Market
- Revenue (KES)
- Yield (Kg)
- Planting Date

---

### Section A: EXACT Function

1. Create a measure that checks if the Crop Type is exactly "Maize" (case-sensitive).
2. Create a measure that verifies whether the Farmer Name is exactly "Mary Wanjiku".
3. Create a column that returns TRUE if County is exactly "Nakuru" and FALSE otherwise.

---

### Section B: FIND Function

1. Create a calculated column that finds the position of the word "Pot" in Crop Type.
2. Write a DAX formula that searches for the word "John" in Farmer Name and returns 0 if not found.
3. Find the position of the substring "Market" inside the Market column.

---

### Section C: FORMAT Function

1. Create a measure that formats total revenue as currency.
2. Format the Planting Date column to display in the format: `March 10, 2023`
3. Create a column that formats Yield (Kg) to two decimal places.

---

### Section D: LEFT Function

1. Create a column that extracts the first 3 letters of Crop Type.
2. Extract the first 4 characters from the County name.
3. Extract the first 5 characters from Farmer Name.

---

### Section E: RIGHT Function

1. Extract the last 3 letters from Crop Type.
2. Extract the last 4 characters from the Market name.
3. Extract the last 6 characters from Farmer Name.

---

### Section F: LEN Function

1. Create a column that calculates the length of the Crop Type name.
2. Find the number of characters in each Farmer Name.
3. Determine the length of the County name.

---

### Section G: LOWER Function

1. Convert Crop Type into lowercase.
2. Convert Farmer Name into lowercase.
3. Convert County names into lowercase.

---

### Section H: UPPER Function

1. Convert Crop Type into uppercase.
2. Convert Farmer Name into uppercase.
3. Convert Market names into uppercase.

---

### Section I: TRIM Function

1. Create a column that removes extra spaces in Crop Type.
2. Remove leading or trailing spaces in Farmer Name.
3. Clean the County column by removing extra spaces.

---

### Section J: CONCATENATE Function

1. Combine Crop Type and Farmer Name into one column.
2. Combine County and Market into one text field.
3. Create a column that combines Farmer Name + Crop Type.

   **Example Output:** `Mary Wanjiku - Potatoes`

---

### Bonus Practice (Real Analysis)

1. Create a column showing First 3 letters of Crop Type + County.

   **Example:** `MAI - Nakuru`

2. Create a column combining Farmer Name + Planting Date.

   **Example:** `John Kamau planted on March 10, 2023`

3. Convert all Farmer Names to uppercase and remove extra spaces.
```


---


## RAG In-Class Activity

**Source file:** `in_class_activity.md`  
**Role in guide:** Retrieval-augmented generation activity using policy documents, PDF extraction, chunking, embeddings, ChromaDB, and evaluation.


## In-Class Activity: Policy Document RAG System Using ChromaDB

### Objective

Build a working RAG system that answers questions from Kenyan policy documents using ChromaDB.

---

### 1. Data Requirement

**Choose one category:**

- Kenya Constitution
- Finance Bills
- Economic Surveys
- County Development Plans
- Parliamentary Acts

**Minimum requirements:**

- At least 1 PDF
- At least 50 total pages

---

### 2. Implementation Tasks

#### Step 1 — Extract Text

**Use:** `pypdf`

Create a function to extract text from PDFs.

**Explain:**

- Why PDFs must be converted to text
- Challenges in text extraction

---

#### Step 2 — Chunking

**Use:** `RecursiveCharacterTextSplitter`

**Parameters:**

- `chunk_size`: 800–1000
- `chunk_overlap`: 150–200

**Explain:**

- Why chunking is necessary
- Why overlap improves retrieval

---

#### Step 3 — Embeddings

**Use:** `BAAI/bge-small-en-v1.5`

Generate normalized embeddings for all chunks.

**Explain:**

- What embeddings represent
- Why semantic search is better than keyword search

---

#### Step 4 — Store in ChromaDB

Create a persistent collection.

**Store:**

- `ids`
- `documents`
- `embeddings`
- `metadata` (source filename)

**Explain:**

- Why metadata is important
- What happens if embeddings are not stored

---

#### Step 5 — Query & Retrieval

- Embed user query
- Retrieve top 3 results
- Display retrieved text and similarity score

**Explain:**

- What similarity score means
- Why lower cosine distance = better match

---

### 3. Evaluation (100 Marks)

| Component           | Marks |
|---------------------|-------|
| Extraction          | 15    |
| Chunking            | 20    |
| Embeddings          | 20    |
| ChromaDB Storage    | 20    |
| Query & Retrieval   | 15    |
| Explanations        | 10    |
| **Total**           | **100** |

---

> **Submission:** Working code + written explanations.


---


## Adaptive Learning Capstone

**Source file:** `CHO5-PROJECT.md`  
**Role in guide:** Full-stack AI capstone using Groq, Aiven PostgreSQL, FastAPI, SQL analytics, and adaptive feedback loops.


## AI-Powered Adaptive Learning System
#### A Beginner-Friendly Project Guide | Groq + Aiven PostgreSQL + FastAPI

---

### 1. Project Overview

You are going to build a learning platform that adjusts to each student. The system will remember what a student got wrong, ask an AI to help explain it, and store everything in a real cloud database.

Before writing any code, think about these questions:

- What information do you need to store about a student?
- What should happen the moment a student submits a wrong answer?
- How would a teacher know which student needs help?

---

### 2. Technology Stack

You will use four tools. Research each one before you start.

| Layer | Tool | What to find out |
|---|---|---|
| Backend | FastAPI | How do you create a route that accepts data? |
| AI | Groq API | How do you send a prompt and read the response? |
| Database | Aiven PostgreSQL | How do you connect using a connection string? |
| Analytics | SQL | What does GROUP BY and HAVING do? |

---

### 3. Database Design

You need to design tables before writing any code. Think about what data each table should hold.

#### Table 1: Students

Think about what basic information identifies a student. Every table needs a way to tell rows apart from each other. What column makes each student unique?

Hint: think about id, name, email, and when they joined.

#### Table 2: Quiz Results

Every time a student answers a question, you want to save that moment. What details matter about that moment?

Hint: think about which student, which topic, what they answered, whether it was right, how long it took, and what the AI said back.

#### Table 3: Performance History

This table is a summary. It does not store individual answers. It stores a rolled-up picture of how a student is doing over time.

Hint: think about averages and totals per topic per student.

---

### 4. Setting Up Your Project

#### Folder Structure

Before writing code, plan your files. A good structure separates concerns: one file for database, one for AI, one for routes.

Hint: think about what belongs together. Database connection code should not live in the same file as your AI calls.

#### Environment Variables

You should never put passwords or API keys directly in your code. Use a separate file to store secrets.

Hint: look up what a `.env` file is and how `python-dotenv` reads it. Your database URL and Groq key go here.

#### Packages to Install

You need four packages. Think about what each one does before installing it.

Hint: one for your web server, one for your database connection, one for Groq, one for reading your `.env` file.

---

### 5. Building the Backend

#### Connecting to Aiven PostgreSQL

Your database connection needs the URL from your Aiven dashboard. The URL contains your host, port, username, password, and database name all in one string.

Hint: look up how `asyncpg.connect()` works. Think about what happens if the connection fails.

#### Endpoint 1: Submit an Answer

This is the most important endpoint. When a student submits an answer, several things need to happen in the right order.

Think about the order:

1. Check if the answer is correct
2. Ask the AI for feedback
3. Save everything to the database
4. Return the result to the student

Hint: the endpoint receives data from the student. What shape should that data be? Look up Pydantic `BaseModel`.

#### Endpoint 2: Student Progress

A teacher or student should be able to see how a student is performing across all topics.

Hint: this endpoint reads from the database, not writes. Think about which SQL keywords group results by topic and calculate averages.

---

### 6. Connecting Groq AI

#### How Groq Works

Groq takes a text prompt and returns a text response. Your job is to write a good prompt that gives Groq enough context to respond helpfully.

Hint: think about what information Groq needs. It cannot see your database. You have to pass the topic, the question, the student's answer, and the correct answer inside the prompt text.

#### Writing Two Different Prompts

The feedback should be different depending on whether the student got it right or wrong.

For a wrong answer, think about what a good teacher would do:
- Would they just say "wrong"?
- Or would they explain, show an example, and give another chance?

For a correct answer, think about what keeps a student motivated and challenged.

Hint: use an if/else to build different prompts for each case.

#### Generating New Questions

You also want Groq to generate quiz questions on any topic at any difficulty level.

Hint: think about what information you need to pass in, and what format you want Groq to respond in. You may want to tell Groq exactly how to structure its response.

---

### 7. SQL Analytics

This is where your system becomes useful for teachers. You will write queries that summarize student performance.

#### Query 1: Find Struggling Students

You want to find students whose average score is below a certain threshold.

Hint: you need to join two tables, group by student, calculate an average, and then filter groups using a keyword that filters after grouping (not WHERE).

#### Query 2: Find the Hardest Topic

You want to find which topic has the lowest average score across all students.

Hint: group by topic, calculate the average score, and sort the results so the lowest comes first.

#### Query 3: Individual Student Report

A student wants to see their own performance broken down by topic.

Hint: filter by student, group by topic, and show the average score and average time taken.

#### Exposing Analytics as an Endpoint

Once your queries work, wrap them in a FastAPI endpoint so teachers can access the data through the API.

Hint: think about what a teacher would want to pass in as a parameter. Maybe a score threshold? Maybe a student id?

---

### 8. Real-Time Feedback Flow

This is the full loop. When it all works together, the sequence looks like this:

```
Student submits answer
        |
        v
Backend checks: correct or incorrect?
        |
        v
Build a prompt using the context
        |
        v
Send prompt to Groq
        |
        v
Groq returns feedback
        |
        v
Save result + feedback to PostgreSQL
        |
        v
Return feedback to student instantly
```

Think about what could go wrong at each step. What happens if Groq is slow? What happens if the database save fails? Good code handles these cases.

---

### 9. Project Phases

Work through this one phase at a time. Do not move to the next phase until the current one works.

| Phase | Focus | You know it works when... |
|---|---|---|
| 1 | Database + API | You can submit an answer and see it saved in Aiven |
| 2 | Groq Integration | Wrong answers return a helpful AI explanation |
| 3 | Analytics | You can query which students are struggling |
| 4 | Real-Time Feedback | All steps happen in one request, end to end |

---

### 10. Questions to Guide Your Thinking

Use these questions as checkpoints as you build:

- Why do we store ai_feedback in the database instead of just showing it once?
- What would happen if two students submit answers at the exact same time?
- How would you change the system to support multiple teachers?
- What SQL query would tell you which student has improved the most over time?
- Why should secrets like API keys never be committed to GitHub?

---

### 11. Running Your Application

Once your code is written, you need to start the server and test it.

Hint: FastAPI uses a command-line tool called `uvicorn` to run. Look up the basic command. FastAPI also automatically creates a page where you can test your endpoints in the browser without writing any extra code.

Think about: how would you test your `/submit-answer` endpoint before building a frontend?

---

*Build it phase by phase. Test each piece before moving forward. The goal is not perfect code — it is a working system you understand.*


---


## Japan Car Import Advisory Capstone

**Source file:** `CapstoneProject - japan-car-price-prediction-project.md`  
**Role in guide:** End-to-end data science product capstone for car price prediction, advisory recommendations, and deployment.


#### **Project: Japan Car Import Advisory Platform**

Build a data engineering and machine learning platform that helps Kenyan car buyers compare the cost of importing cars from Japan with buying cars locally.

Extract car listing data for vehicles manufactured from **2018 onwards** from platforms such as:

- **SBT Japan** — `www.sbtjapan.com`
- **Car From Japan** — `carfromjapan.com`
- **AAAJapan** — `aaajapan.com`
- **JapaneseCarTrade / JCT** — `www.japanesecartrade.com`
- **BE FORWARD** — `www.beforward.jp`

Store the extracted data in a database of your choice, clean it, and prepare it for analysis.

Using the cleaned data, build a platform that allows users to estimate the full cost of importing a car into Kenya, including:

- Purchase price
- Shipping cost
- KRA taxes
- Port charges
- Clearing fees
- Registration costs
- Any other relevant import charges

The system should also compare the estimated import cost with local market prices and show the potential savings from importing.

Additionally, build a machine learning model that predicts car prices in Japan based on features such as:

- Make
- Model
- Year of manufacture
- Mileage
- Engine size
- Fuel type
- Transmission type
- Body type
- Source platform

The final solution should include:

- A database
- Cleaned dataset
- Data extraction scripts
- Import cost calculator
- Machine learning model
- Dashboard or web application
- Clear project documentation
- Final presentation


---



## Appendix: Curation Checklist

Use this checklist before delivering the guide to a cohort:

- [ ] Confirm the cohort calendar and decide whether the program will run for 6 months or extend toward 9 months.
- [ ] Verify that all learners can open Excel, Power BI Desktop, a SQL environment, Python, Git, and a code editor.
- [ ] Assign the Kenya Crops dataset early and reuse it across tools to reinforce transfer learning.
- [ ] Require every major artifact to include a problem statement, assumptions, cleaning notes, findings, and next steps.
- [ ] Review dashboards for visual hierarchy, readable labels, correct aggregation, and responsible use of filters.
- [ ] Review notebooks for reproducibility, clear section headings, controlled random seeds where appropriate, and meaningful evaluation.
- [ ] Review AI/RAG projects for retrieval quality, source attribution, metadata, failure cases, and responsible limitations.
- [ ] Require a final portfolio walkthrough that connects technical work to business or social impact.