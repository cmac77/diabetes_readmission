## src/modeling/xgboost_model_pipeline.py

# Summary of Folders and Expected Files
# This section outlines where outputs and logs will be stored when the pipeline is executed.
# - results/logs/: Contains model training logs (e.g., model_training.log).
# - results/models/: Stores the trained model (e.g., model_logistic_regression_xgboost.pkl).
# - results/reports/: Contains performance reports (e.g., report_logistic_regression_xgboost.txt).
# - results/figures/: Includes SHAP analysis figures for feature importance and dependence plots.

#%% Base Imports
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
from pyprojroot import here
import cupy as cp  # For GPU acceleration
import joblib
import matplotlib.pyplot as plt
import shap
import seaborn as sns

#%% Local Imports
# Setup the root path for project directory
path_root = Path(here())
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

from src.config import load_config

#%% Logging Setup
# Configure logging to store model training logs
path_logs = path_root / "results" / "logs"
path_logs.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=path_logs / "model_training.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("XGBoost Model Pipeline Initialized")

#%% Helper Methods
def ensure_directory(path):
    """Ensures the specified directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)

def save_figure(figure, path, title):
    """Saves a figure to the given path and logs the action."""
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)  # Close figure to prevent memory issues
    logging.info(f"{title} saved to {path}")

#%% XGBoost Model Pipeline Class
class XGBoostMulticlassPipeline:
    """
    A pipeline for training, evaluating, and analyzing an XGBoost multiclass model.
    
    This class addresses the following target classes:
    - 0: Patient not readmitted
    - 1: Patient readmitted in less than 30 days
    - 2: Patient readmitted in more than 30 days
    
    Attributes:
        use_gpu (bool): Enable GPU acceleration with CuPy if set to True.
        target_column (str): The name of the target variable in the dataset.
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
    """
    
    def __init__(self, use_gpu=True, target_column="readmitted", hyperparameter_tuning=False):
        """Initialize the pipeline with configuration, paths, and model setup."""
        # Load configurations from the YAML file
        self.config = load_config()
        self.use_gpu = use_gpu
        self.target_column = target_column
        self.hyperparameter_tuning = hyperparameter_tuning

        # Define paths for storing data and results
        self.path_root = path_root
        self.path_data_cleaned = self.path_root / self.config['paths']['data_cleaned']
        self.path_data_transformed = self.path_root / self.config['paths']['data_transformed']
        self.path_models = self.path_root / self.config['paths']['results'] / "models"
        self.path_reports = self.path_root / self.config['paths']['results'] / "reports"

        # Ensure results directories exist
        self.path_models.mkdir(parents=True, exist_ok=True)
        self.path_reports.mkdir(parents=True, exist_ok=True)

        # Configure device for GPU/CPU usage and update model parameters
        device = 'cuda' if use_gpu else 'cpu'
        xgb_params = self.config.get("params_xgboost", {}).copy()
        xgb_params.pop("hyperparameter_tuning", None)  # Remove tuning params from model config
        xgb_params.update({
            "tree_method": "hist",  # Use histogram-based method for efficient training
            "enable_categorical": use_gpu,  # Enable categorical handling if using GPU
            "device": device,
            "eval_metric": "mlogloss"  # Multi-class log loss metric
        })

        # Initialize XGBoost model and feature scaler
        self.model = XGBClassifier(**xgb_params)
        self.scaler = StandardScaler()
        logging.info("Model and Scaler initialized")

    def load_data(self):
        """Load preprocessed data from a pickle file."""
        logging.info("Loading data...")
        return pd.read_pickle(self.path_data_cleaned)

    def calculate_class_weights(self, y):
        """Calculate class weights to handle class imbalance."""
        logging.info("Calculating class weights...")
        classes = np.unique(y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    def feature_engineering(self, df, oversampling_method="ADASYN"):
        """
        Perform feature engineering and optional class balancing using oversampling methods.
        
        Args:
            df (DataFrame): Input data.
            oversampling_method (str): The oversampling technique to use ("SMOTE" or "ADASYN").
        
        Returns:
            tuple: Feature matrix and target variable.
        """
        logging.info("Starting feature engineering...")

        # Separate target variable
        if self.target_column in df.columns:
            ser_target = df.pop(self.target_column)
        else:
            raise KeyError(f"Target column '{self.target_column}' not found in DataFrame.")

        # Identify numerical and categorical columns
        list_numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        list_categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Encode categorical features for non-GPU processing
        if not self.use_gpu:
            logging.info("Encoding categorical features...")
            for col in list_categorical_cols:
                df[col] = df[col].astype(str).fillna("missing")
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])

        # Scale numerical features
        df[list_numerical_cols] = self.scaler.fit_transform(df[list_numerical_cols])

        # Create interaction features to capture relationships between predictors
        logging.info("Creating interaction features...")
        df['number_inpatient_x_diabetesMed'] = df['number_inpatient'] * df['diabetesMed']
        df['number_inpatient_x_number_emergency'] = df['number_inpatient'] * df['number_emergency']
        df['number_inpatient_x_change'] = df['number_inpatient'] * df['change']
        df['diabetesMed_x_number_emergency'] = df['diabetesMed'] * df['number_emergency']

        # Apply log transformations to selected features
        logging.info("Applying log transformations...")
        for col in ['number_inpatient', 'number_emergency', 'time_in_hospital']:
            if (df[col] > 0).all():  # Ensure values are positive
                df[f"log_{col}"] = np.log(df[col])

        # Apply oversampling to balance classes
        logging.info(f"Applying {oversampling_method} for class balancing...")
        if oversampling_method == "SMOTE":
            smote = SMOTE(random_state=42)
            df, ser_target = smote.fit_resample(df, ser_target)
        elif oversampling_method == "ADASYN":
            adasyn = ADASYN(random_state=42)
            df, ser_target = adasyn.fit_resample(df, ser_target)
        else:
            raise ValueError("Invalid oversampling method. Use 'SMOTE' or 'ADASYN'.")

        logging.info("Feature engineering complete")
        return df, ser_target

    def analyze_feature_importance(self, feature_names):
        logging.info("Analyzing feature importance...")
        importances = self.model.feature_importances_
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_feature_names = [feature_names[i] for i in sorted_idx]

        # Plot the feature importances
        plt.figure(figsize=(12, 6))
        plt.barh(sorted_feature_names[:20], sorted_importances[:20])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Name")
        plt.title("Top 20 Feature Importances")
        plt.gca().invert_yaxis()
        plt.show()

        logging.info("Feature importance analysis complete")

    def shap_analysis(self, X, y, sample_size=1000):
        logging.info("Performing SHAP analysis with stratified sampling...")

        # Ensure the figures directory exists
        path_figures = self.path_root / "results" / "figures"
        ensure_directory(path_figures)

        # Stratified sampling to get a representative sample
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size / len(y), random_state=42)
        for train_index, sample_index in splitter.split(X, y):
            X_sample = X.iloc[sample_index]
            y_sample = y.iloc[sample_index]

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        for class_index in range(shap_values.shape[2]):  # Loop over each class
            shap_values_for_class = shap_values[:, :, class_index]

            # Bar Plot
            bar_figure = plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values_for_class, X_sample, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance (Bar Plot) - Class {class_index}")
            save_figure(bar_figure, path_figures / f"shap_feature_importance_class_{class_index}_bar.png",
                        f"SHAP Feature Importance Bar Plot for Class {class_index}")
            plt.close(bar_figure)  # Close the bar plot figure

            # Summary Plot
            summary_figure = plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values_for_class, X_sample, show=False)
            plt.title(f"SHAP Feature Impact (Summary Plot) - Class {class_index}")
            save_figure(summary_figure, path_figures / f"shap_feature_impact_class_{class_index}_summary.png",
                        f"SHAP Feature Impact Summary Plot for Class {class_index}")
            plt.close(summary_figure)  # Close the summary plot figure

            # Dependence Plots
            top_features = X_sample.columns[:5]  # Adjust number of features as needed
            for feature in top_features:
                if feature in X_sample.columns:
                    try:
                        # Explicitly create and close the dependence plot
                        shap.dependence_plot(
                            feature, shap_values_for_class, X_sample, interaction_index=None, show=False
                        )
                        plt.title(f"SHAP Dependence Plot for {feature} - Class {class_index}")
                        plt.savefig(
                            path_figures / f"shap_dependence_plot_{feature}_class_{class_index}.png", bbox_inches='tight'
                        )
                        plt.close('all')  # Close all figures to prevent memory issues
                    except Exception as e:
                        logging.error(f"Error plotting SHAP dependence plot for feature {feature} and class {class_index}: {e}")
                        plt.close('all')  # Ensure all figures are closed even on error

        logging.info("SHAP analysis complete with stratified sampling")

    def train_model(self, X, y):
        logging.info("Training model...")
        class_weights = self.calculate_class_weights(y)
        print("Class Weights: ", class_weights)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if self.use_gpu:
            X_train = cp.array(X_train)
            X_test = cp.array(X_test)
            y_train = cp.array(y_train)
            y_test = cp.array(y_test)
        else:
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values

        # Perform hyperparameter tuning if enabled
        if self.hyperparameter_tuning:
            self.hyperparameter_tuning_function(X_train, y_train)

        self.model.fit(X_train, y_train)
        logging.info("Model training complete")

        y_pred = self.model.predict(X_test)
        if self.use_gpu:
            y_test = y_test.get()
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Model Accuracy: {accuracy:.2f}")
        logging.info("Classification Report:\n" + report)

        print(f"XGBoost Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", report)

        # Call analyze_class_2_errors to analyze misclassifications for class 2
        self.analyze_class_2_errors(X_test, y_test, y_pred)

        self.save_outputs(report)

        return accuracy, report
    
    def analyze_class_2_errors(self, X_test, y_test, y_pred):
        logging.info("Analyzing misclassifications for class 2...")

        # Generate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        class_labels = ["Not Readmitted", "Readmitted < 30 Days", "Readmitted > 30 Days"]

        # Save the confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(self.path_root / "results" / "figures" / "confusion_matrix.png", bbox_inches='tight')
        plt.close()  # Close the figure to prevent rendering

        # Focus on misclassifications of class 2
        misclassified_as = y_pred[(y_test == 2) & (y_pred != 2)]

        # Create a table showing how often class 2 was misclassified as each other class
        error_distribution = pd.Series(misclassified_as).value_counts().rename_axis("Misclassified As").reset_index(name="Count")
        print("\nClass 2 Misclassification Distribution:")
        print(error_distribution)

        # Save the bar chart for misclassification distribution
        plt.figure(figsize=(6, 4))
        sns.barplot(x="Misclassified As", y="Count", data=error_distribution)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Class 2 Misclassification Distribution")
        plt.savefig(self.path_root / "results" / "figures" / "class_2_misclassification_distribution.png", bbox_inches='tight')
        plt.close()  # Close the figure to prevent rendering

        logging.info("Class 2 misclassification analysis complete")

    def hyperparameter_tuning_function(self, X_train, y_train):
        logging.info("Starting hyperparameter tuning...")
        # Access hyperparameter tuning settings from the nested config
        tuning_config = self.config['params_xgboost']['hyperparameter_tuning']
        param_grid = tuning_config['param_grid']
        cv = tuning_config['cv']
        scoring = tuning_config['scoring']
        n_iter = tuning_config['n_iter']

        #    
        tuner = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=42
        )
        tuner.fit(X_train, y_train)
        self.model = tuner.best_estimator_  # Update model with the best parameters
        logging.info("Hyperparameter tuning complete")

    def save_outputs(self, report):
        logging.info("Saving outputs...")
        path_model_file = self.path_models / "model_logistic_regression_xgboost.pkl"
        path_report_file = self.path_reports / "report_logistic_regression_xgboost.txt"

        joblib.dump(self.model, path_model_file)
        with open(path_report_file, "w") as file:
            file.write("Classification Report:\n")
            file.write(report)

        logging.info(f"Model saved to {path_model_file}")
        logging.info(f"Classification report saved to {path_report_file}")


#%% Main Execution
if __name__ == "__main__":
    logging.info("Executing main pipeline...")
    pipeline = XGBoostMulticlassPipeline(use_gpu=False, hyperparameter_tuning=False)
    df = pipeline.load_data()
    df_features, ser_target = pipeline.feature_engineering(df)
    accuracy, report = pipeline.train_model(df_features, ser_target)

    # Perform SHAP analysis on the sample
    pipeline.shap_analysis(df_features, ser_target)
    logging.info("Pipeline execution complete")
#%%