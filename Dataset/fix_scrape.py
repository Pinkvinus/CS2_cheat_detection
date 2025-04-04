import pandas as pd

# Sample DataFrame
df = pd.read_csv("Demo_data/cs_scrape_2025-04-01 19_38_27.339191.csv")



# Convert to a string representation of a list
df['cheater_names_str'] = df['cheater_names_str'].apply(lambda x: x.split('-') if isinstance(x, str) else '[]')

# Print the modified DataFrame

df.to_csv("cs_scrape_2025_04_03.csv")