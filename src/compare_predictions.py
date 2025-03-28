import pandas as pd

def compare_predictions(file_x, file_y, output_file):
    # Load the files into dataframes
    df_x = pd.read_csv(file_x)
    df_y = pd.read_csv(file_y)
    
    # Ensure both dataframes have the 'llm_pred' column
    if 'llm_pred' not in df_x.columns or 'llm_pred' not in df_y.columns:
        raise ValueError("Both files must contain the 'llm_pred' column")
    
    # Create a new dataframe to store the comparison results
    comparison_df = pd.DataFrame({
        'class': df_x['Classification'],
        'pred1': df_x['llm_pred'],
        'pred2': df_y['llm_pred'],
        'equals': (df_x['llm_pred'] == df_y['llm_pred']) & (df_x['llm_pred'] == df_x['Classification']),
        'Summary': df_x.get('Summary', pd.NA),
        'Description': df_x.get('Description', pd.NA),
        'Enviroment': df_x.get('Enviroment', pd.NA)
    })
    
    # Save the comparison results to the output file
    comparison_df.to_csv(output_file, index=False)

# Example usage
if __name__ == "__main__":
    compare_predictions('output/4cd401b5f5799f64ef72777d558f2456/results.csv', 'output/aaa5b434d3698140776bd9dc84071f22/results.csv', 'output/comparison_results.csv')