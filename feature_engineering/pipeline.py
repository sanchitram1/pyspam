from dependency_offline import handle_dependency
from description_offline import handle_description
from handle_null import fill_null
from load_data import load_json
from maintainer_offline import handle_maintainers
from name_based_offline import add_name_based
from remove_redundant import drop_redundant
from save_json import save_json
from settings import INPUT_PATH, OUTPUT_PATH
from time_offline import handle_time


def main(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    df, legit_mask = load_json(input_path)
    df = add_name_based(df, legit_mask_np=legit_mask)
    df = handle_description(df, legit_mask)
    df = handle_maintainers(df)
    df = handle_dependency(df)
    df = handle_time(df)
    df = drop_redundant(df)
    df = fill_null(df)
    save_json(df, output_path)


if __name__ == "__main__":
    file_path = "data/bq-results-20251207-053959-1765086112714.json"
    main(input_path=file_path)
