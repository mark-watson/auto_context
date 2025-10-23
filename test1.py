from auto_context import AutoContextPersistent

if __name__ == '__main__':
    # Initialize the AutoContext system with the data directory
    print("\n--- Initializing AutoContext... ---")
    auto_context: AutoContextPersistent = AutoContextPersistent(directory_path="data")

    # Test query 1: Economics related question
    my_query: str = "who said the study of economics is bullshit?"
    
    # Retrieve relevant context for the query (limiting to 1 result)
    augmented_prompt: str = auto_context.get_prompt(my_query, num_results=1)

    # Display the generated prompt for the economics question
    print("\n\n\n\n--- Generated Prompt for who said the study of economics is bullshit? ---")
    print(augmented_prompt)

    # Test query 2: Chemistry related question
    second_query: str = "Talk about glassware used in chemistry"
    # Retrieve relevant context for the query (limiting to 3 results)
    second_prompt: str = auto_context.get_prompt(second_query, num_results=3)
    # Display the generated prompt for the chemistry question
    print("\n\n\n\n--- Generated Prompt for Talk about glassware used in chemistry ---")
    print(second_prompt)

    # Test query 3: Health/psychology related question
    third_query: str = "Talk about not liking open spaces"
    # Retrieve relevant context for the query (limiting to 2 results)
    third_prompt: str = auto_context.get_prompt(third_query, num_results=2)
    # Display the generated prompt for the open spaces question
    print("\n\n\n\n--- Generated Prompt for Talk about not liking open spaces ---")
    print(third_prompt)
