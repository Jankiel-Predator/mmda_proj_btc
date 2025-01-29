import pandas as pd

def preprocess_bitcoin_data(input_file: str, output_file_path: str = 'bitcoin_preprocessed.csv'):
    # Load the dataset
    bitcoin = pd.read_csv(input_file)
    
    # Drop unnecessary column
    if 'Unnamed: 0' in bitcoin.columns:
        bitcoin = bitcoin.drop(columns=['Unnamed: 0'])
    
    # Convert numerical columns (excluding 'Date' and last two columns)
    for name in bitcoin.columns[1:-2]:
        bitcoin[name] = bitcoin[name].str.replace(',', '', regex=True).astype(float)
    
    # Convert Date format and sort by Date
    bitcoin['Date'] = pd.to_datetime(bitcoin['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    bitcoin = bitcoin.sort_values('Date')
    
    # Add Average column
    bitcoin['Average'] = (bitcoin['High'] + bitcoin['Low']) / 2
    
    # Convert Date column back to datetime
    bitcoin['Date'] = pd.to_datetime(bitcoin['Date'])
    
    # Save the preprocessed data
    bitcoin.to_csv(output_file_path, index=False)
    
    print(f"Preprocessed data saved to {output_file_path}")