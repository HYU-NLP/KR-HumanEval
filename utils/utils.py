from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import List, Union, Iterable, Dict
import numpy as np
import random
import gzip
import json
import torch
import itertools
import tqdm
import os

from utils.execution import check_correctness

ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "HumanEval.jsonl")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    
def read_problems(task_name) -> Dict[str, Dict]:
    base_directory = os.path.dirname(os.path.abspath(__file__))
    test_set_file = os.path.join(base_directory, "..", "data", f"{task_name}.jsonl")
    return {task["task_id"]: task for task in stream_jsonl(test_set_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                # skip empty line
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)): # read sample one by one
            task_id = sample["task_id"] # classify each sampling given task_id
            completion = sample["completion"] # single string
            is_total_function = sample["is_total_function"]
            print(type(is_total_function))
            args = (problems[task_id], completion, timeout, completion_id[task_id], is_total_function)
            future = executor.submit(check_correctness, *args) # evaluate correctness one by one and aggregate
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        # for sample in tqdm.tqdm(stream_jsonl(sample_file)): # read sample one by one
        #     task_id = sample["task_id"] # classify each sampling given task_id
        #     for completion in sample['completions']:
        #         args = (problems[task_id], completion, timeout, completion_id[task_id], is_chat_model)
        #         future = executor.submit(check_correctness, *args) # evaluate correctness one by one and aggregate
        #         futures.append(future)
        #         completion_id[task_id] += 1
        #         n_samples += 1
            
        assert len(completion_id) == len(problems), "Some problems are not attempted."

        # print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)): # as_completed yields futures as they complete
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # after finishing evaluating all samples,
    # results = {'HumanEval/0': [(0, result0),,, (n, resultn)],
    #            'HumanEval/1': [(0, result0),,, (n, resultn)],...}
    # where n is the number of samples,
    # and result is a dictionary of 'task_id', 'completion', 'passed', 'completion_id'

    # calculate pass@k for each problem and average.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result] # r = (n, resultn)
        total.append(len(passed)) # number of sampling per problem
        correct.append(sum(passed)) # number of correct sample per problem
    total = np.array(total) # ex) [10, 10, ,,, 10] (num_sample is 10, len(total) = # total problem)
    correct = np.array(correct) # ex) [2, 1, ,,, 3] correct answer per each problem

    # calculate pass rate per problem and average => pass@k
    # and repeat for every k
    # if n=10, pass_at_k = {'pass@1': v1, 'pass@10': v2}
    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    
    # Finally, save the results in one file:
    # the number of sample yielded by this function is the same to the number of line of 'sample_file'
    # that is, yield as many samples as you have generated
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0) # (n, resultn) is popped from results
            # sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    # print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k