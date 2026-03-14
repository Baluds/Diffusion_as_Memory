from bert_score import score
import json

def compute_bert_score(src_list, output_list):
    # can try bigger models like "microsoft/deberta-xlarge-mnli" for better performance, but it will be slower.
    P, R, F1 = score(
        src_list,
        output_list,
        lang="en",
        verbose=True
    )
    print("Precision:", P)
    print("Recall:", R)
    print("F1:", F1)
    print("Mean F1:", F1.mean().item())

    results = {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
        "mean_f1": F1.mean().item()
    }

    # save to file
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=4)


def get_src_and_output():
    with open("output/p0/inference/test_preds.json", "r") as f:
        data = json.load(f)
    src_list = []
    output_list = []
    for item in data:
        candidate = item["x_true"]
        src_list.append(candidate)
        reference = item["x_pred"]
        output_list.append(reference)
    
    return src_list, output_list


if __name__ == "__main__":
    src_list, output_list = get_src_and_output()
    compute_bert_score(src_list, output_list)