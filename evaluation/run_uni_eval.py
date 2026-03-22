import sys
import json
import os
import importlib.util

# change this your local path to UniEval.
# where you have cloned the UniEval repository: git clone https://github.com/maszhongming/UniEval.git
UNIEVAL_PATH = "/project/pi_dagarwal_umass_edu/project_3/shared/UniEval"
if not os.path.isdir(UNIEVAL_PATH):
    raise FileNotFoundError(f"UniEval path not found: {UNIEVAL_PATH}")
if UNIEVAL_PATH not in sys.path:
    sys.path.insert(0, UNIEVAL_PATH)


def _load_unieval_helpers(unieval_path: str = UNIEVAL_PATH):
    """Load UniEval helpers from the shared UniEval repo explicitly by path."""
    utils_py = os.path.join(unieval_path, "utils.py")
    spec = importlib.util.spec_from_file_location("unieval_utils", utils_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load UniEval utils module from {utils_py}")
    unieval_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unieval_utils)

    # metric.evaluator is inside UniEval; ensure that repo root is at the front.
    if unieval_path not in sys.path:
        sys.path.insert(0, unieval_path)
    from metric.evaluator import get_evaluator  # type: ignore

    return unieval_utils.convert_to_json, get_evaluator


def evaluate_factual_consistency(src_list, output_list):
    convert_to_json, get_evaluator = _load_unieval_helpers()
    task = 'fact'
    data = convert_to_json(output_list=output_list, src_list=src_list)
    evaluator = get_evaluator(task, device='cpu')
    results = evaluator.evaluate(data, print_result=True)

    all_results = [result["consistency"] for result in results]
    mean = sum(all_results) / len(all_results)
    print("Mean Consistency:", mean)
    print(results)

    log_results = {
        "mean_consistency": mean,
        "results": results
    }

    # save to file
    with open("unieval_scores.json", "w") as f:
        json.dump(log_results, f, indent=4)


def evaluate_factual_consistency_return(src_list, output_list):
    convert_to_json, get_evaluator = _load_unieval_helpers()
    task = 'fact'
    data = convert_to_json(output_list=output_list, src_list=src_list)
    evaluator = get_evaluator(task, device='cpu')
    results = evaluator.evaluate(data, print_result=True)

    all_results = [result["consistency"] for result in results]
    mean = sum(all_results) / len(all_results)
    print("Mean Consistency:", mean)
    print(results)

    log_results = {
        "mean_consistency": mean,
        "results": results
    }

    return log_results


def get_src_and_output(file_path, ground_label_key="x_true", prediction_key="x_pred"):
    with open(file_path, "r") as f:
        data = json.load(f)
    src_list = []
    output_list = []
    for item in data:
        candidate = item[ground_label_key]
        src_list.append(candidate)
        reference = item[prediction_key]
        output_list.append(reference)
    
    return src_list, output_list


if __name__ == "__main__":
    src_list, output_list = get_src_and_output("output/p0/inference/test_preds.json", ground_label_key="x_true", prediction_key="x_pred")
    evaluate_factual_consistency(src_list, output_list)

