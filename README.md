# Support Vector Machine (SVM) Hyperparameter Tuning and Model Optimization

This project demonstrates how to use **Support Vector Machine (SVM)** for classification tasks, focusing on the optimization of the **SVM hyperparameters** using **GridSearchCV** for better performance. The dataset used is the **Mushroom dataset** from the UCI Machine Learning Repository.

## Overview

In this project, we:
1. **Load the Mushroom dataset** and preprocess the data.
2. Perform **Label Encoding** to convert categorical variables into numeric format.
3. Split the dataset into **training** and **testing** sets.
4. **Scale the features** using `StandardScaler` to normalize the data.
5. Use **GridSearchCV** to find the best hyperparameters for the **SVM model**.
6. Train the SVM model using the optimized parameters and evaluate its performance.
7. Generate a **convergence graph** for tracking the model's accuracy over iterations.

## Requirements

To run this project, you will need the following Python packages:

- `pandas`: For data manipulation and analysis
- `numpy`: For numerical computations
- `scikit-learn`: For machine learning models and preprocessing
- `matplotlib`: For plotting the convergence graph

## Results
Best Hyperparameters: The best C, kernel, and gamma values found by GridSearchCV will be printed.
Model Accuracy: The accuracy of the optimized SVM model on the test set will be displayed.
Convergence Graph: A plot of the SVM model's convergence over 100 iterations (simulated in this case) will be shown.(The graph has been uploaded in the repo)

## Conclusion
This project demonstrates the use of SVM for classification tasks and the importance of hyperparameter tuning to optimize model performance. By using GridSearchCV, we can identify the best parameters for the model, improving accuracy and generalization.

## Future Work
Improve the Convergence Graph: Track and plot the accuracy over real iterations during training.
Experiment with other models: Compare SVM with other classification models like Random Forest or K-Nearest Neighbors.
