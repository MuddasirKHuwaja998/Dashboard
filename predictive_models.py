import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

def predictive_models_page(df, target_columns):
    # Check if target columns exist in the dataframe
    missing_columns = [col for col in target_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing target columns: {', '.join(missing_columns)}")

    # Handle missing values in the dataset
    st.write("Handling missing values...")
    
    # Fill missing values in the features with the median (for numerical data)
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=target_columns)), columns=df.drop(columns=target_columns).columns)
    
    # Fill missing values in the target columns with the mode (most frequent value)
    for target_column in target_columns:
        df_imputed[target_column] = df[target_column].fillna(df[target_column].mode()[0])

    # Perform prediction for each target column
    st.subheader("Predictive Modeling")

    for target_column in target_columns:
        st.write(f"Training model for {target_column}...")

        # Prepare features and target
        X = df_imputed.drop(columns=target_columns)
        y = df_imputed[target_column]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the model (RandomForest as an example)
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Display results
        st.write(f"Model for {target_column} - Accuracy: {accuracy:.2f}")
        st.text(f"Classification Report for {target_column}:\n{classification_rep}")

        # Show the true vs predicted values
        predictions_df = pd.DataFrame({
            'True Values': y_test,
            'Predicted Values': y_pred
        })

        st.subheader(f"True vs Predicted values for {target_column}")
        st.write(predictions_df)

        # Feature importance (optional)
        st.write(f"Feature Importance for predicting {target_column}:")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.write(feature_importance)
