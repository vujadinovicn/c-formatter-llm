import clang.cindex

def save_dataset_to_json(dataset, filename):
    """Save the dataset to a JSON file."""
    dataset.to_json(filename)

def get_number_of_tokens(node):
    return len(list(node.get_tokens()))

def get_tokens_and_spacing(node, code):
    """Get token spellings, kinds, and spacing information from the parsed node."""
    tokens = list(node.get_tokens())
    token_kinds = []
    token_spellings = []
    spaces = []

    for i, token in enumerate(tokens):
        token_kinds.append(str(token.kind))
        token_spellings.append(token.spelling)

        if i < len(tokens) - 1:
            end_of_current = token.extent.end
            start_of_next = tokens[i + 1].extent.start
            between = code[end_of_current.offset:start_of_next.offset]
            spaces.append(1 if ' ' in between else 0)

    return token_spellings, token_kinds, spaces

def initialize_clang_library():
    """Initialize the Clang library for token analysis."""
    library_path = ".\\venv\\lib\\site-packages\\clang\\native\\libclang.dll"
    clang.cindex.Config.set_library_file(library_path)
    return clang.cindex.Index.create()