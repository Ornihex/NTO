from pathlib import Path
import torch





# КОНСТАНТЫ

class constants:
    # --- FILENAMES ---
    TRAIN_FILENAME = "train.csv"
    TEST_FILENAME = "test.csv"
    USER_DATA_FILENAME = "users.csv"
    BOOK_DATA_FILENAME = "books.csv"
    BOOK_GENRES_FILENAME = "book_genres.csv"
    GENRES_FILENAME = "genres.csv"
    BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
    SUBMISSION_FILENAME = "submission.csv"
    TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
    BERT_EMBEDDINGS_FILENAME = "bert_embeddings.pkl"
    BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
    PROCESSED_DATA_FILENAME = "processed_features.parquet"

    # --- COLUMN NAMES ---
    # Main columns
    COL_USER_ID = "user_id"
    COL_BOOK_ID = "book_id"
    COL_TARGET = "rating"
    COL_SOURCE = "source"
    COL_PREDICTION = "rating_predict"
    COL_HAS_READ = "has_read"
    COL_TIMESTAMP = "timestamp"

    # Feature columns (newly created)
    F_USER_MEAN_RATING = "user_mean_rating"
    F_USER_RATINGS_COUNT = "user_ratings_count"
    F_BOOK_MEAN_RATING = "book_mean_rating"
    F_BOOK_RATINGS_COUNT = "book_ratings_count"
    F_AUTHOR_MEAN_RATING = "author_mean_rating"
    F_BOOK_GENRES_COUNT = "book_genres_count"

    # Metadata columns from raw data
    COL_GENDER = "gender"
    COL_AGE = "age"
    COL_AUTHOR_ID = "author_id"
    COL_PUBLICATION_YEAR = "publication_year"
    COL_LANGUAGE = "language"
    COL_PUBLISHER = "publisher"
    COL_AVG_RATING = "avg_rating"
    COL_GENRE_ID = "genre_id"
    COL_DESCRIPTION = "description"

    # --- VALUES ---
    VAL_SOURCE_TRAIN = "train"
    VAL_SOURCE_TEST = "test"

    # --- MAGIC NUMBERS ---
    MISSING_CAT_VALUE = "-1"
    MISSING_NUM_VALUE = -1
    PREDICTION_MIN_VALUE = 0
    PREDICTION_MAX_VALUE = 10
    
    

# КОНФИГ

class config:
    # --- DIRECTORIES ---
    # Настраиваем пути для работы в текущей директории (Colab/Kaggle)
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = ROOT_DIR / "output"
    MODEL_DIR = OUTPUT_DIR / "models"
    SUBMISSION_DIR = OUTPUT_DIR / "submissions"

    # --- PARAMETERS ---
    RANDOM_STATE = 67
    TARGET = constants.COL_TARGET

    # --- TEMPORAL SPLIT CONFIG ---
    # 0.8 means 80% of data points (by timestamp) go to train, 20% to validation
    TEMPORAL_SPLIT_RATIO = 0.8

    # --- TRAINING CONFIG ---
    EARLY_STOPPING_ROUNDS = 150
    MODEL_FILENAME = "lgb_model.txt"

    # --- TF-IDF PARAMETERS ---
    TFIDF_MAX_FEATURES = 1000
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    TFIDF_NGRAM_RANGE = (1, 2)

    # --- BERT PARAMETERS ---
    BERT_MODEL_NAME = constants.BERT_MODEL_NAME
    BERT_BATCH_SIZE = 8
    BERT_MAX_LENGTH = 512
    BERT_EMBEDDING_DIM = 768
    BERT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BERT_GPU_MEMORY_FRACTION = 0.90

    # --- FEATURES ---
    CAT_FEATURES = [
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
        constants.COL_GENDER,
        constants.COL_AGE,
        constants.COL_AUTHOR_ID,
        constants.COL_PUBLICATION_YEAR,
        constants.COL_LANGUAGE,
        constants.COL_PUBLISHER,
    ]

    # --- MODEL PARAMETERS ---
    LGB_PARAMS = {
        "objective": "rmse",
        "metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "n_jobs": -1,
        "seed": RANDOM_STATE,
        "boosting_type": "gbdt",
    }

    LGB_FIT_PARAMS = {
        "eval_metric": "rmse",
        "callbacks": [],
    }