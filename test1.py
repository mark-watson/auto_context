from auto_context import AutoContext

if __name__ == '__main__':
    print("\n--- Initializing AutoContext... ---")
    auto_context = AutoContext(directory_path="../data")

    my_query = "who said the study of economics is bullshit?"
    
    augmented_prompt = auto_context.get_prompt(my_query, num_results=1)

    print("\n\n\n\n--- Generated Prompt for who said the study of economics is bullshit? ---")
    print(augmented_prompt)

    second_query = "Talk about glassware used in chemistry"
    second_prompt = auto_context.get_prompt(second_query, num_results=3)
    print("\n\n\n\n--- Generated Prompt for ATalk about glassware used in chemistry---")
    print(second_prompt)

    third_query = "Talk about not liking open spaces"
    third_prompt = auto_context.get_prompt(third_query, num_results=2)
    print("\n\n\n\n--- Generated Prompt for Talk about not liking open spaces ---")
    print(third_prompt)
