## Credit Risk Classification with Logistic Regression

This project aims to classify credit risk using a logistic regression model on an imbalanced dataset. The dataset, which consists of historical lending activity, will be split into training and testing sets and evaluated with both the original and resampled data using the `RandomOverSampler` module from the imbalanced-learn library. The performance of the model will be measured using balanced accuracy score, confusion matrix, and classification report. The results will be summarized in a credit risk analysis report included in the GitHub repository's `report.md` file.

## Technologies

This project leverages the following technologies:
* [Python 3.7.13](https://www.python.org/downloads/release/python-385/) - The programming language used in the project.
* [Pandas](https://pandas.pydata.org/) - A Python library used for efficient data manipulation.
* [Jupyter Lab](https://jupyter.org/) - An interactive development environment used to create and share documents containing live code, equations, visualizations, and narrative text.
* [Numpy](https://numpy.org/) - A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
* [Matplotlib](https://matplotlib.org/) - A plotting library for the Python programming language.
* [Scikit-learn 1.0.2](https://scikit-learn.org/stable/index.html) - A Python library containing efficient tools for machine learning and statistical modeling, including classification, regression, clustering, and dimensionality reduction.
* [Imbalanced-learn](https://imbalanced-learn.org/stable/index.html) - A Python module to tackle the curse of imbalanced datasets.

## Installation

To set up the development environment for this project, you will need to install the required libraries and technologies. Here are the steps to do so:

1. Install Python: This project requires Python 3.7.13 or higher. You can download the latest version of Python from the official Python website: [Python 3.7.13](https://www.python.org/downloads/release/python-385/)

2. Create a virtual environment: It is recommended to use a virtual environment to isolate the dependencies for this project. You can use the following command to create a virtual environment:

```
python -m venv myenv
```
3. Activate the virtual environment: 
* To activate the virtual environment, use the following command (for Windows):
```
myenv\Scripts\activate
```
* Or the following command (for macOS and Linux):

```
source myenv/bin/activate
```
4. Install the required packages: The required packages for this project are listed in the requirements.txt file. To install these packages, you can use the following command:
```
pip install -r requirements.txt
```
5. Launch Jupyter Lab: To launch Jupyter Lab, you can use the following command:
```
jupyter lab
```
6. Open the project file: Once Jupyter Lab is open, navigate to the location where you have cloned this project and open the **Credit_Risk_Classification.ipynb** file .

### Note:
If you are running on a Mac with M1 or M2 chip, there may be issues with installing TensorFlow. In this case, please refer to this link for troubleshooting: https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/

## Instructions:

This challenge consists of the following subsections:

* Split the Data into Training and Testing Sets

* Create a Logistic Regression Model with the Original Data

* Predict a Logistic Regression Model with Resampled Training Data 

### Split the Data into Training and Testing Sets

Open the starter code notebook and then use it to complete the following steps.

1. Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

2. Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

    > **Note** A value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.  

3. Check the balance of the labels variable (`y`) by using the `value_counts` function.

4. Split the data into training and testing datasets by using `train_test_split`.

### Create a Logistic Regression Model with the Original Data

Employ your knowledge of logistic regression to complete the following steps:

1. Fit a logistic regression model by using the training data (`X_train` and `y_train`).

2. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.

4. Answer the following question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

### Predict a Logistic Regression Model with Resampled Training Data

Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use `RandomOverSampler`.

To do so, complete the following steps:

1. Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

2. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.

3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.
    
4. Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

### Write a Credit Risk Analysis Report

For this section, you’ll write a brief report that includes a summary and an analysis of the performance of both machine learning models that you used in this challenge. You should write this report as the `README.md` file included in your GitHub repository.

Structure your report by using the report template that `Starter_Code.zip` includes, and make sure that it contains the following:

1. An overview of the analysis: Explain the purpose of this analysis.


2. The results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.

3. A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.

### Contributors
Alexander Likhachev

### License
MIT