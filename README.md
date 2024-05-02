# Fraud Detection classification
This project by Keith Chen Yong, aimed at fraud detection for bank transactions. <br>

## Project Title
Financial Fraud Detection Using Classification Algorithms

##  Introduction
This initiative aims to predict the fraudulent transactions using synthetic financial datasets sourced from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1/data). Correctly identifying whether a transaction is fraudulent is extremely important as fraudsters aim to empty out the victim's bank account through methods that involve transferring and then immediately cashing out. If fraud occurs and suspicious transactions are not stopped, it can result in life-changing scenarios. Correctly identifying fraudulent transactions will improve the customer's perception of the bank. The core task of this project involves the application of classification algorithms to facilitate this prediction. Initially, three algorithms are employed: *Logistic Regression*, *Random Forest Classification*, and *XGBoost*. These were selected for their proven effectiveness in similar tasks, with the structure of the code designed to accommodate the introduction of additional algorithms or hyperparameter fine-tuning of existing ones for optimal performance.

## Installation
All the necessary dependencies are specified in requirements.txt.

## Usage
Upon pushing or manually triggering the workflow in GitHub, the necessary installations from requirements.txt and the execution of the run.sh bash script will be completed. This will be followed by the execution of the main.py script. This Python script sequentially triggers various Python files responsible for tasks such as data ingestion (from a .csv file within a folder titled data), data preprocessing, model building, k-fold cross-validation, and lastly, evaluation of the algorithms. Users will be able to see the k-fold cross-validation results, offering a comprehensive view of the dataset's behavior under different models and various classification performance metrics for each algorithm. <br><br>
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/1cde79bc-0552-4c53-b230-6c9786eaa16d)


User will be able to see the k-fold cross-validation results, offering a holistic view of the dataset's behavior under different models and various classification Performance metrics for each algorithm.<br>


## Configuration
Users have the flexibility to modify the following
1. Modify the k value in config.ini for cross-validation
2. Fine-tuning hyperparameters in config.ini to improve model performance or tailor it to different datasets
3. Introduce new algorithms in config.ini for exploring additional analytical approaches. *Note: Additional of new algorithms will require modify of model.py. Please contact author for further guidance, contact information listed below*

| ML Algorithm            | Justification |
|-------------------------|---------------|
| Logistic Regression     | - Simple, fast, and efficient for linearly separable data.<br>- Good interpretability |
| Random Forest Classifier | - Handles non-linear data well due to ensemble of decision trees.<br>- Robust to overfitting with large number of features.
| XGBoost | - Handling Imbalanced Data<br>-  Boosting Technique |

## Contributing
Contributions are welcome! Please submit bug reports or feature requests via the [GitHub issue tracker](https://github.com/KeithChenYong/Fraud-Detection/issues).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details. 

## Credits
**Kaggle - Edgar Lopez-Rojas (Author)** as the [source](https://www.kaggle.com/datasets/ealaxi/paysim1) of data provision. A synthetic dataset is generated using a simulator called PaySim due to a lack of publicity available dataset due to the intrinsically private nature.

## Contact
Author Name:   Chen Yong
Author Email:  keith.chenyong@gmail.com
For questions or feedback, please contact [me](mailto:keith.chenyong@gmail.com).

## Additional Documentation
For guidance on hyperparameter optimization, please refer to the following link.
1. [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
2. [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier)
3. [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html)

## Choosen machine learning algorithms
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost

## Folder overview
The folder contains a mix of directories and files designed to organize the project's codebase, data, documentation, and operational workflows. This structure facilitates easy navigation and understanding of the project's purpose, how to set it up, and how it operates.

<u>Folder Structure</u>
1. .github: This directory houses GitHub Actions workflows. These workflows are automated processes that can be set to run on specific triggers, such as pushing code to the repository or manually triggering them. 
2. data: The data directory is intended to contain datasets used by the project. 
3. src: This folder contains 7 the Python scripts associated with the project. These scripts perform a variety of tasks, including the main script, data injestion, data preprocessing, cross validating, model training and performance evaluation
4. LICENSE.txt: This file contains the licensing information for the project. The license dictates how others can use, modify, and distribute the project. 
5. README.md: The README.md file is a Markdown file used to provide an overview of the project. It includes information on what the project does, how to set it up, how to use it, and any other relevant information that a user or contributor might need to know.
6. eda.ipynb: This Jupyter notebook contains exploratory data analysis (EDA). The notebook are used to explore and visualize the data, understand its structure, and derive insights that can guide further analysis and modeling.
7. requirements.txt: The requirements.txt file lists all the Python dependencies required to run the project. 
8. run.sh: This is a shell script file that contains commands to run the project.
9. config.ini: Contains various editable parameters for cross validation and algorithms
10. .gitattributes: Enables large files to be stored in Github

## EDA Overview
The interactive notebook provides a detailed analysis of the synthetic financial dataset generated by PaySim. The dataset includes two target columns: 'isFlaggedFraud' and 'isFraud', with 16 and 8,213 rows out of a total of 6,362,620 entries. Financial fraud is typically represented by a minority class in datasets.<br>
1. 'isFlaggedFraud' are descripted as an illegal attempt in this dataset to transfer more than 200,000 in a single transaction
2. 'isFraud' are descripted as fraudster attempts at taking control of customers' accounts and trying to empty the funds by transferring to another account and then cashing out of the system <br>
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/5ef84d2a-666d-442a-9a9f-7b829480f681)

<br>Further analysis suggests that it is almost impossible to distinguish between 'isFlaggedFraud' and 'isFraud,' even with additional filtering. The total number of 'isFlaggedFraud' instances based on the new criteria is 2,732, instead of the actual 16. Fortunately, 'isFraud' successfully identifies transactions that would have been marked as 'isFlaggedFraud.' Due to the limitations of the dataset's features, 'isFlaggedFraud' was decided to be removed.

'isFraud,' on the other hand, aligns with the described scenario. By filtering 'TRANSFER' followed by 'CASH_OUT' and ensuring the 'amount' is the same between 'TRANSFER' and 'CASH_OUT,' we were able to identify all 8,213 fraudulent transactions. Feature engineering was applied based on this scenario.

A second feature engineering attempt utilized the balance columns. By calculating the difference between new and old balances, we found that fraud is more prevalent among wealthy customers, most of whom have more than 1,000,000 in their originating balance. <br>
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/f399b7ba-1dc4-476c-a255-e62d0f2c6616)

Data partitioning was done using a customized stratified sampling algorithm. Its principle involves undersampling the non-target class data.
1. Divide the available data into two sets (strata):
  - all samples of the class of interest (set A).
  - all other samples (set B).
2. Construct the training set:
  - randomly select 50% of the samples in set A.
  - add an equal number of samples from set B.
3. Construct the test set:
  - select the remaining 50% of samples from set A.
  - add enough samples from set B to restore the original ratio from the overall data set.
<br>
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/20f2d738-63ec-4675-8a1b-73b6931bfdf0)

Interpreting the feature importances/coefficient for XGBoost, Logistic Regression, and Random Forest showed that both engineered columns 'emptied' and 'wealthy_customer' significantly contributed to the algorithms. <br>
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/81757781-be6b-403b-8ea6-92f57aaf41f0)<br>
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/9558ec77-e307-4789-b156-8194a27c8fc2)<br>
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/a2b27343-d10d-497c-967b-d020cbdb8b02)<br>

<br>Logistic Regression and XGBoost performed the best compared to Random Forest.
![image](https://github.com/KeithChenYong/Fraud-Detection/assets/133010489/33d90f99-c3b8-4ffe-8b8e-516ed75aeb61)


