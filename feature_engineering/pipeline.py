from load_data import load_json
from name_based_offline import add_name_based
from description_offline import handle_description
from maintainer_offline import handle_maintainers
from dependency_offline import handle_dependency
from time_offline import handle_time
from remove_redundant import drop_redundant
from handle_null import fill_null
from save_json import save_json
from settings import INPUT_PATH, OUTPUT_PATH


def main(input_path = INPUT_PATH, output_path = OUTPUT_PATH):
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
    file_path = "/Users/reezhan/Desktop/UCB/242A/project/spam/pyspam_features.jsonl"
    main(input_path=file_path)