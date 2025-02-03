import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class BitcoinPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        
        # Drop unnecessary column
        if 'Unnamed: 0' in X.columns:
            X = X.drop(columns=['Unnamed: 0'])
        
        # Convert numerical columns (excluding 'Date' and last two columns)
        for name in X.columns[1:-2]:
            X[name] = X[name].str.replace(',', '', regex=True).astype(float)
        
        # Convert Date format and sort by Date
        X['Date'] = pd.to_datetime(X['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
        X = X.sort_values('Date')
        
        # Add Average column
        X['Average'] = (X['High'] + X['Low']) / 2
        
        # Convert Date column back to datetime
        X['Date'] = pd.to_datetime(X['Date'])
        
        return X


# Create a pipeline
bitcoin_pipeline = Pipeline([
    ('preprocessing', BitcoinPreprocessor())
])


df = pd.read_csv("data/bitcoin/bitcoin_original.csv")
processed_df = bitcoin_pipeline.fit_transform(df)
processed_df.to_csv("bitcoin_preprocessed.csv", index=False)
