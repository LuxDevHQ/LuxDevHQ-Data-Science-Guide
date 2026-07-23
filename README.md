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
