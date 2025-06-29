# Data Science and Analytics Curriculum (Excel, Power BI, SQL, Python)
 

**Mode of Delivery:**  Physical and Online (Live Sessions & Hands-on Projects)

---
 
## Week 1: Microsoft Excel for Data Analysis
 
### **Day 1: Excel Basics & Navigation**
- What is Excel and why it's important
- Excel interface: ribbons, sheets, cells, rows, columns
- Data types: text, numbers, dates
- Entering and editing data
- Basic formatting: font, color, alignment, number formatting
- Saving and organizing files
- Keyboard shortcuts
 
### **Day 2: Working with Formulas and Functions**
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
 
### **Day 3: Data Cleaning & Sorting/Filtering**
- Remove duplicates
- Find & Replace
- Text to Columns
- Error checking: ISERROR(), IFERROR()
- Sorting data (A-Z, Z-A, custom sort)
- Filtering data (AutoFilter, Custom Filter)
- Freeze panes, hide/unhide rows/columns
 
### **Day 4: Data Analysis with Pivot Tables, Charts, and Dashboards**
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
 
## Week 2: Power BI for Data Visualization
---
 
## **Day 1: Introduction to Power BI and Power Query Editor (Data Transformation)**
 
### Overview & Setup
- What is Power BI? (Comparison with Excel & Tableau)
- Power BI Components: Desktop, Service, Mobile
- Installing Power BI Desktop
- Power BI Desktop Interface Overview
 
### Getting Data
- Importing Data from Excel, CSV, Web
- Understanding Data Types and Field Formatting
 
### Power Query Editor
- Opening Power Query Editor
- Removing Rows, Columns, and Duplicates
- Changing Data Types, Renaming Columns
- Splitting & Merging Columns
- Using “Replace Values”
- Applied Steps, Reordering, and Removing Steps
 
### Combining Queries
- **Merge Queries** (SQL-style joins)
- **Append Queries** (Union of datasets)
 
---
 
## **Day 2: DAX Basics – Measures, Calculated Columns & Aggregations**
 
### Introduction to DAX
- What is DAX (Data Analysis Expressions)?
- Syntax Rules: =, (), [], Table & Column references
 
### Calculated Columns vs Measures
- When to use each
- Creating new fields using Calculated Columns
- Building Measures for aggregations
 
### Common DAX Functions
- **Aggregation**: SUM(), AVERAGE(), COUNT(), COUNTROWS(), DISTINCTCOUNT()
- **Logical**: IF(), SWITCH(), AND(), OR()
- **Date**: TODAY(), NOW(), YEAR(), MONTH(), DATEDIFF()
- **Text**: CONCATENATE(), LEFT(), RIGHT(), LEN()
 
### Practical Use Cases
- Total Revenue, Profit Margin, % Growth
- IF(Sales > 50000, “High”, “Low”)
- Number of Orders per Customer
 
---
 
## **Day 3: Data Modeling, Relationships & Joins**
 
### Data Modeling Concepts
- What is a Data Model?
- Star Schema vs Snowflake Schema
- Importance of Fact and Dimension tables
 
### Relationships in Power BI
- One-to-Many and Many-to-One relationships
- Creating & Managing Relationships in Model View
- Active vs Inactive Relationships
 
### Data Joins
- Relationship-based vs Merge Query joins
- Cardinality (One-to-One, One-to-Many)
- Cross filter direction
 
### Modeling Best Practices
- Hiding unnecessary columns
- Using Lookup Tables
- Creating Role-Playing Dimensions (e.g. Order Date vs Delivery Date)
 
---
 
## **Day 4: Visualizations, Charts & Dashboards**
 
### Basic Visuals
- Bar Chart, Column Chart, Line Chart
- Pie & Donut Charts
- Card, KPI
- Table & Matrix
 
### Advanced Visuals
- Tree Map, Funnel, Gauge
- Maps: Filled Map, Shape Map, ArcGIS Map
- Custom Visuals (via AppSource)
 
### Interactivity
- Visual-Level, Page-Level, Report-Level Filters
- Slicers (Text, Date, Dropdowns)
- Drill-down & Drill-through
- Tooltips, Bookmarks, Buttons
 
### Dashboard Building
- Designing a complete report page
- Adding Titles, Backgrounds, Logos, Images
- Aligning and Formatting Visuals
- Creating Navigation Buttons
 
### Publishing & Sharing
- Publishing to Power BI Service
- Sharing Reports & Dashboards
- Setting Scheduled Refresh
 
---
 
 
## Week 3: SQL for Data Analysis
 
### **Day 1   Introduction to SQL, Table creation and manipulation and SQL KEY words  **
- What is SQL, relational databases, and DBMS
- CREATE, DROP, ALTER TABLE
- INSERT, UPDATE, DELETE data
- SELECT and FROM
- ORDER BY ASC/DESC
- LIMIT and OFFSET
- GROUP BY and HAVING
 
### **Day 2  Aggregations and operators**
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
 
### **Day 3: SQL Joins and Relationships**
- Primary and foreign keys
- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- FULL OUTER JOIN
- CROSS JOIN
- Self Join and aliasing
- Combining joins with aggregations and filters
- Practice: build reports using joins from multiple tables
 
 
### **Day 4: Advanced SQL (CTEs, Subqueries, Stored Procedures)**
- Subqueries: scalar, correlated, IN/EXISTS
- Common Table Expressions (CTEs)
- Stored Procedures and parameters (Intro)
 
 
---
 
##  Week 4: Python for Data Analysis
 
### **Day 1: Python Basics**
- What is Python and its use in data
- Setting up Python (Jupyter, Colab, VSCode)
- Variables and data types (int, float, string, bool)
- Lists, Tuples, Dictionaries, Sets
- Operators and expressions
- Conditional statements: if, elif, else
- Loops: for, while
- Functions and basic I/O
 
### **Day 2: Working with Data using Pandas**
- Importing data: CSV, Excel
- DataFrames and Series
- Inspecting data: head(), tail(), info(), describe()
- Indexing and selection: loc[], iloc[], slicing
- Filtering data
- Sorting, renaming columns
- Adding, deleting columns
 
### **Day 3: Data Cleaning and Analysis**
- Handling missing data: isnull(), fillna(), dropna()
- Duplicates: duplicated(), drop_duplicates()
- Changing data types
- GroupBy and Aggregations
- Merging and joining DataFrames
- Pivot tables with pandas
- Basic visualizations: matplotlib, seaborn
 
### **Day 4: Projects and Automation with Python**
- Real-world mini project (sales data, HR data, etc.)
- Creating summary reports
- Exporting cleaned data to Excel/CSV
- Automating tasks with loops and functions
- Intro to working with APIs
- Final recap and practice challenges
 
---
 


# 3-Week Machine Learning & Deep Learning Course Plan

---

## Week 1: Supervised Learning

---

### [Day 1: Introduction to Machine Learning & Supervised Learning](https://github.com/LuxDevHQ/Introduction_to_ML_basic_concepts.git)
**Topic:** What is Machine Learning? Overview of Supervised Learning  
**Summary:**
- Introduction to ML and real-world applications
- Types of ML: Supervised vs. Unsupervised vs. Reinforcement
- Concept of datasets: features, labels, training vs. testing data
- Supervised learning definition and flow
- Use cases of classification and regression

---

### Day 2: [Regression Algorithms](https://github.com/LuxDevHQ/Introduction-to-linear-regression.git)
**Topic:** [Linear Regression](https://github.com/LuxDevHQ/simple-linear-regression.git)
**Summary:**
- Understanding continuous outputs
 **Topic:** [Multiple Linear Regression](https://github.com/LuxDevHQ/multiple-linear-regression.git)
- Linear regression: line of best fit, cost function, gradient descent
- Evaluation metrics: MSE, RMSE, R² score
- Hands-on example (e.g., predicting house prices)
**Topic** [Lasso And Ridge Regression](https://github.com/LuxDevHQ/Lasso_and_rRdge_regression.git)

---

### Day 3: [Classification Algorithms](https://github.com/LuxDevHQ/Introduction_to_classification.git)
**Topic:**[Logistic Regression](https://github.com/LuxDevHQ/logistic_regression_notes.git)
**Summary:**
- Understanding classification problems
- Logistic regression intuition and sigmoid function
- Decision boundaries
**Topic** [K-Nearest Neighbors (KNN)](https://github.com/LuxDevHQ/knn_notes.git)  
- KNN: how it works, choosing K
- Evaluation: accuracy, confusion matrix, precision, recall, F1 score

---

###  [Day 4: Introduction To Ensemble Models](https://github.com/LuxDevHQ/Introduction_to_Ensemble_Methods.git)
**Topic:** [Deep Dive Into Ensemble Models](https://github.com/LuxDevHQ/Deep_dive_into_ensemble_methods.git) 
**Summary:**
- Decision Trees: how they split data, overfitting
- Random Forest: ensemble of trees, bagging
- Gradient Boosting & XGBoost: boosting techniques
- Bias-variance tradeoff
  Get decision trees notes [here](https://github.com/LuxDevHQ/Decision_trees_notes.git)

---

## Week 2: Unsupervised Learning + Model Evaluation

---

### [Day 1: Introduction to Unservised learning](https://github.com/LuxDevHQ/Introduction_to_Unsupervised_Learning.git)
**Topic:** [Clustering and K-Means](https://github.com/LuxDevHQ/Clustering_and_KMeans_In_Unsupervised_learning.git)
**Summary:**
- What is clustering? Applications
- K-means: centroids, choosing K, inertia
- Elbow method for optimal clusters
  **Topic** [Hierarchical clustering](https://github.com/LuxDevHQ/Herarchial_clustering.git)
- dendrograms
- Hands-on with customer segmentation

---

### [Day 2: Dimensionality Reduction](https://github.com/LuxDevHQ/PCA.git)
**Topic:** PCA and t-SNE  
**Summary:**
- Curse of dimensionality
- Principal Component Analysis (PCA): reducing dimensions while preserving variance
- t-SNE for visualizing high-dimensional data
- Use cases: visualization and speed-up training

---

### [Day 3: Model Evaluation & Validation Techniques](https://github.com/LuxDevHQ/Model_evaluation_and_validation_summary.git)
**Topic:** Cross-validation and Performance Metrics  
**Summary:**
- Why train/test split isn’t enough
- K-fold cross-validation
- Evaluation metrics recap (for both classification and regression)
- ROC, AUC, Precision-Recall curves
- Avoiding overfitting/underfitting

---

### [Day 4: Feature Engineering & Data Preprocessing](https://github.com/LuxDevHQ/Feature_Engineering.git)
**Topic:** Cleaning, Encoding, Scaling, Feature Selection  
**Summary:**
- Handling missing data
- Encoding categorical variables (Label Encoding, One-Hot)
- Feature scaling (Normalization, Standardization)
- Feature selection techniques (correlation, chi-square, recursive elimination)

---

## Week 3: Hyperparameter Tuning + Real-World Applications

---

### [Day 1: Hyperparameter Tuning_Introduction](https://github.com/LuxDevHQ/Hyperparameter_tuning_introduction.git)
**Topic:** Grid Search, Random Search, and Bayes Optimization  
**Summary:**
- Difference between parameters and hyperparameters
- Grid Search and Randomized Search using sklearn
- Bayesian optimization with optuna or hyperopt
- Hands-on with hyperparameter tuning in Random Forest / XGBoost
- [Deep dive into hyper-parameter tuning](https://github.com/LuxDevHQ/Deep_dive_into_hyperparameter_tuning.git)

---

### [Day 2: Model Pipelines and Deployment Basics](https://github.com/LuxDevHQ/Model_pipelining_and_deployment.git)
**Topic:** Building End-to-End Pipelines  
**Summary:**
- Using Pipeline and ColumnTransformer from sklearn
- Combining preprocessing and model steps
- Saving/loading models with joblib or pickle
- Introduction to deployment (streamlit / flask APIs)

---

### Day 3: Capstone Project Work Session
**Topic:** Work on Individual or Group ML Projects  
**Summary:**
- Students select a dataset
- Apply full ML pipeline: EDA, preprocessing, modeling, evaluation
- Option to use classification or regression
- Instructor guides students through challenges

---

### Day 4: Capstone Project Presentations + Review
**Topic:** Final Presentations and Course Recap  
**Summary:**
- Each student/team presents their project
- Q&A and peer feedback
- Recap of everything covered
- Tips for further learning: books, courses, Kaggle, real-world practice

---

#  Week 1: Foundations of Deep Learning & ANN Basics

---

### [Day 1: Introduction to Deep Learning](https://github.com/LuxDevHQ/Introduction_to_deep_learning.git)
**Topic:** What is Deep Learning? Understanding Neural Networks  
**Summary:**
- Difference between machine learning and deep learning
- Structure of a biological vs. artificial neuron
- Anatomy of a neural network (layers, nodes, weights, biases)
- Activation functions (ReLU, sigmoid, tanh)
- Forward pass intuition

---

### [Day 2: Building a Neural Network from Scratch](https://github.com/LuxDevHQ/Building_Neural_networls_and_Application_of_Deepl_Learning.git)
**Topic:** Feedforward Neural Networks  
**Summary:**
- Forward propagation in multi-layer networks
- Loss functions (MSE, Cross-Entropy)
- Backpropagation and gradient descent (basic idea, not full math)
- Training loop overview
- Implementing a simple NN

---

### [Day 3: Deep Learning with TensorFlow/Keras](https://github.com/LuxDevHQ/Deep_learning_with_tensorflow_keras.git)
**Topic:** Implementing Neural Networks with Keras  
**Summary:**
- Introduction to TensorFlow and Keras
- Building a neural network model using Keras
- Compiling and fitting a model
- Understanding epochs, batch size, learning rate
- Evaluating models and making predictions

---

### [Day 4: Improving Model Performance](https://github.com/LuxDevHQ/Improving-_model_performance.git)
**Topic:** Regularization and Optimization  
**Summary:**
- Overfitting and underfitting in deep networks
- Regularization techniques: Dropout, L1/L2 penalties
- Optimizers: SGD, Adam, RMSprop
- Hands-on: tuning models using callbacks, early stopping, etc.

---

## Week 2: Deep Diving into ANNs

---

### [Day 1: Model Design – MLPs, Sequential vs Functional API](https://github.com/LuxDevHQ/Functional_and_Sequential_API.git)
**Topic:** Multi-layer Perceptrons (MLPs) and Keras APIs  
**Summary:**
- MLP architecture and the role of depth in networks
- Using Sequential API for simple stack-like models
- Using Functional API for complex models (multiple inputs/outputs, shared layers, skip connections)
- Hands-on: Building models with both APIs for the same task
- Comparison of readability, flexibility, and real-world use cases

---

### [Day 2: Handling Different Data Types](https://github.com/LuxDevHQ/Hanling_Different_Datatypes.git)
**Topic:** Image, Tabular, and Text Data with ANNs  
**Summary:**
- Preprocessing images for ANN (flattening, normalization)
- Handling categorical and numerical tabular data
- Introduction to tokenizing and embedding text
- Practical: building ANN for tabular and basic image data

---

### [Day 3: Model Evaluation and Visualization](https://github.com/LuxDevHQ/Model_Evaluation_and_Vizualisation.git)
**Topic:** Evaluating Deep Learning Models  
**Summary:**
- Accuracy, confusion matrix, precision, recall, F1 score
- ROC curves and AUC
- Visualizing training history: loss vs. accuracy curves
- Model interpretability basics (Grad-CAM preview, attention maps)

---

### Day 4: Capstone 1 – ANN Mini Project
**Topic:** Project using ANN on real dataset  
**Summary:**
- Students use datasets (e.g., MNIST, tabular UCI data, sentiment classification)
- Build, tune, and evaluate ANN models
- Start-to-finish pipeline implementation
- Prepare short presentations

---

# Week 3: CNNs for Image Data

---

### [Day 1: Introduction to CNNs](https://github.com/LuxDevHQ/Introduction_to_CNN.git)
**Topic:** Convolutional Neural Networks Basics  
**Summary:**
- Limitations of ANNs with image data
- Convolutions, filters/kernels, stride, padding
- Max pooling and feature maps
- Architecture overview of CNNs (Conv → Pool → FC)
- CNN vs ANN in performance and structure

---

### [Day 2: Building CNNs in Keras](https://github.com/LuxDevHQ/Building_cnns.git)
**Topic:** Hands-on with CNN for Image Classification  
**Summary:**
- Creating CNN layers with Conv2D, MaxPooling2D, Flatten, Dense
- Training a CNN on MNIST or CIFAR-10
- Data augmentation with ImageDataGenerator
- Improving accuracy using dropout and regularization

---

### [Day 3: Transfer Learning](https://github.com/LuxDevHQ/Transfer_Learning.git)
**Topic:** Pre-trained Models (VGG, ResNet, MobileNet)  
**Summary:**
- What is transfer learning and why it's useful  
- Feature extraction vs. fine-tuning  
- Loading pre-trained models with Keras  
- Customizing top layers for new tasks  
- Hands-on: image classification with MobileNetV2 or VGG16  

---

###  Day 4: Capstone 2 – CNN Project + Course Wrap-up
**Topic:** Final Project + Deep Learning Recap  
**Summary:**
- Students implement an image classifier using CNN or transfer learning
- Full pipeline: preprocessing, training, evaluation
- Project presentations and peer feedback
- Recap of ANN vs CNN, deployment tips, future learning paths (e.g., RNNs, Transformers)
