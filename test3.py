import os
from auto_context import AutoContextPersistent

# Initialize the AutoContextPersistent object with the path to the data directory
auto_context = AutoContextPersistent(directory_path="./data", persist_directory="./chroma_db")

# Define a new test that uses a different prompt
def test_custom_query():
    # Define a custom query
    query = "how did 'sports' get its name?"
    
    # Retrieve the augmented prompt based on the query
    augmented_prompt = auto_context.get_prompt(query, num_results=3)
    
    # Print the generated prompt (or use it in further code)
    print(augmented_prompt)

# Run the test
test_custom_query()
