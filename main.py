from src import preprocess
from src import model
from src import train
from src import evaluate
from src.config import URL, RAW_DATA_DB, RAW_DATA_CSV, CLEANED_DATA_CSV

query = f"SELECT * FROM lung_cancer"

def main():
    try: 
        print("Starting downloading of lung cancer database")
        preprocess.download_db(URL, RAW_DATA_DB)
        print("Lung cancer database downloaded")
    except:
        print("An error occurred while downloading the lung cancer data")

    df = preprocess.query_db(RAW_DATA_DB, query)
    preprocess.save_data(df, RAW_DATA_CSV)
    print("Saved lung cancer data in data folder")

    try:
        print("Preprocessing and feature engineering of raw lung cancer data")
        df = preprocess.preprocess_data(df)
        df = preprocess.feature_engineer(df)
        print("...Done")
    except:
        print("An error occurred while processing")

    try:
        print("Scaling and encoding of processed lung cancer data")
        df = preprocess.standard_scaler(df)
        df = preprocess.encoder(df)
        print("...Done")
    except: 
        print("An error occurred while scaling and encoding data")
    
    preprocess.save_data(df, CLEANED_DATA_CSV)

    print("Select your preferred model for training\n 1. Logistic regression\n 2. Random Forest\n 3. XGBoost")
    model_choice = None
    while model_choice not in ['1', '2', '3']:
        model_choice = input("Enter a choice (1, 2, or 3): ")
        if model_choice not in ['1', '2', '3']:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    if model_choice == '1':
        model_choice = 'Logistic regression'
        choosen_model = model.logistic_regression()
    elif model_choice == '2':
        model_choice = 'Random Forest'
        choosen_model = model.random_forest()
    elif model_choice == '3':
        model_choice == 'XGBoost'
        choosen_model = model.xgboost()
    print(f"You have choosen {model_choice} model")

    print("Choose your preferred porportion for test set.")
    valid_test = 0
    while valid_test == 0:
        test_size = input("Enter preferred test size (0.1 - 0.5): ") 
        try: 
            test_size = float(test_size)
            if test_size >= 0.1 and test_size <= 0.5:
                valid_test = 1
            else: 
                valid_test = 0
                print("Invalid test size. Enter a value between 0.1 and 0.5 (inclusive).")
        except:
            valid_test = 0
            ("Invalid test size. Enter a value between 0.1 and 0.5 (inclusive).")
    
    print(f"You have selected a test size of {test_size}.")
    print("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train.split_data(df, test_size)
    print("...Done")
    print(f"Training {model_choice} model")
    trained_model = train.train(choosen_model, X_train, y_train)
    print("...Done")

    print("Choose your preferred threshold for lung cancer prediction.")
    valid_threshold = 0
    while valid_threshold == 0:
        threshold = input("Enter preferred threshold (0.1 - 0.4): ") 
        try: 
            threshold = float(test_size)
            if threshold >= 0.1 and threshold <= 0.4:
                valid_threshold = 1
            else: 
                valid_threshold = 0
                print("Invalid test size. Enter a value between 0.1 and 0.4 (inclusive).")
        except:
            valid_threshold = 0
            ("Invalid test size. Enter a value between 0.1 and 0.4 (inclusive).")

    print(f"You have selected a threshold of {threshold}.")
    print(f"Evaluating {model_choice} model performance on test set.")
    evaluate.evaluate(trained_model, X_test, y_test, threshold)

if __name__ == '__main__':
    main()

