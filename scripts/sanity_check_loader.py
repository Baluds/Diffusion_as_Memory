# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[1]))

# from src.data.msr_datamodule import MSRGistDataModule

# def main():
#     dm = MSRGistDataModule(
#         train_path="data/processed/train_part2_clean.json",
#         tokenizer_name="bert-base-uncased",
#         batch_size=4,
#         max_length=128,
#         num_workers=0,
#         y_key="y",
#         include_xt=False,
#         make_xplus=True,
#         xplus_max_replacements=2,
#         deterministic_xplus=True,
#     )
#     dm.setup()
#     batch = next(iter(dm.train_dataloader()))

#     print("x_input_ids", batch["x_input_ids"].shape)
#     print("xplus_input_ids", batch["xplus_input_ids"].shape)
#     print("labels", batch["labels"].shape)

#     for i in range(4):
#         print("\n--- sample", i, "---")
#         print("x     :", batch["raw_x"][i])
#         print("x_plus:", batch["raw_x_plus"][i])
#         print("y     :", batch["raw_y"][i])

# if __name__ == "__main__":
#     main()


import os
from src.data.msr_datamodule import MSRGistDataModule

def main():
    # You should export MODEL_DIR before running this (shown in commands below)
    model_dir = os.environ.get("MODEL_DIR", "")
    if not model_dir:
        raise RuntimeError("MODEL_DIR env var not set. Export it to the gemma snapshot dir first.")

    dm = MSRGistDataModule(
        train_path="data/processed/train_part2_clean.json",
        tokenizer_name="bert-base-uncased",
        batch_size=2,
        max_length=128,
        num_workers=0,
        y_key="summary",         # change to "y" if your cleaned file uses y
        include_xt=False,

        make_xplus=True,
        gemma_model_dir=model_dir,
        xplus_cache_jsonl="data/processed/xplus_cache_part2.jsonl",
    )

    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    print("x_input_ids:", batch["x_input_ids"].shape)
    print("xplus_input_ids:", batch["xplus_input_ids"].shape)
    print("labels:", batch["labels"].shape)

    for i in range(batch["x_input_ids"].shape[0]):
        print("\n--- sample", i, "---")
        print("id:", batch["id"][i])
        print("x:", batch["raw_x"][i])
        print("x+:", batch["raw_x_plus"][i])

    # optionally persist new x+ generations from this run
    dm.flush_xplus_cache_to_jsonl()

if __name__ == "__main__":
    main()
