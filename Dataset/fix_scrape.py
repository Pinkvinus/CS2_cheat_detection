import pandas as pd

# Sample DataFrame
df = pd.read_csv("cs_scrape_2025-03-28 17_09_24.042783.csv")



# Convert to a string representation of a list
df['cheater_names_str'] = df['cheater_names_str'].apply(lambda x: x.split('-') if isinstance(x, str) else '[]')

# Print the modified DataFrame
print(df)

df.to_csv("cs_scrape_test_script.csv")