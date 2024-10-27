import clang.cindex

def save_dataset_to_json(dataset, filename):
    """Save the dataset to a JSON file."""
    dataset.to_json(filename)

def get_number_of_tokens(node):
    return len(list(node.get_tokens()))

def initialize_clang_library():
    """Initialize the Clang library for token analysis."""
    library_path = ".\\venv\\lib\\site-packages\\clang\\native\\libclang.dll"
    clang.cindex.Config.set_library_file(library_path)
    return clang.cindex.Index.create()