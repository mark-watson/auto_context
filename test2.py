from auto_context import AutoContextPersistent

if __name__ == '__main__':
    print("\n--- Initializing AutoContext... ---")
    with AutoContextPersistent(directory_path="data") as auto_context:

        my_query: str = "who said the study of economics is bullshit?"
    
        augmented_prompt: str = auto_context.get_prompt(my_query, num_results=1)

        print("\n\n\n\n--- Generated Prompt for who said the study of economics is bullshit? ---")
        print(augmented_prompt)

        second_query: str = "Talk about glassware used in chemistry"
        second_prompt: str = auto_context.get_prompt(second_query, num_results=3)
        print("\n\n\n\n--- Generated Prompt for Talk about glassware used in chemistry ---")
        print(second_prompt)

        third_query: str = "Talk about not liking open spaces"
        third_prompt: str = auto_context.get_prompt(third_query, num_results=2)
        print("\n\n\n\n--- Generated Prompt for Talk about not liking open spaces ---")
        print(third_prompt)
