def parse_input(input_str):
    """Parse --input string into data loading flags.

    Returns (use_features, use_embeddings, embeddings_files).

    Examples::

        "features"                          → (True,  False, [])
        "embeddings_dinov3_vitl"            → (False, True,  ["embeddings_dinov3_vitl.pt"])
        "embeddings_250"                    → (False, True,  ["embeddings_250.pt"])
        "features+embeddings_dinov3_vitl"   → (True,  True,  ["embeddings_dinov3_vitl.pt"])
        "features+embeddings_dinov3_vitl+embeddings_union512" → (True, True, ["embeddings_dinov3_vitl.pt", "embeddings_union512.pt"])
    """
    parts = input_str.split("+")
    use_features = False
    use_embeddings = False
    embeddings_files = []

    for part in parts:
        if part == "features":
            use_features = True
        elif part.startswith("embeddings"):
            use_embeddings = True
            embeddings_files.append(f"{part}.pt")
        else:
            raise ValueError(
                f"Unknown input component: '{part}'. "
                "Expected 'features' or 'embeddings[_variant]'."
            )

    if not use_features and not use_embeddings:
        raise ValueError(
            "--input must include 'features' and/or 'embeddings[_variant]'"
        )

    return use_features, use_embeddings, embeddings_files
