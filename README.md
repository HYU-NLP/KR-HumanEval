## KR_HumanEval Benchmark
The Korean version of the HumanEval benchmark, where the original docstrings are translated into Korean
* Used DeepL translator API for translation
* Performed manual post-processing and LLM-based quality evaluation after translation
* Data format and benchmark performance evaluation method are identical to those of HumanEval.
* Paper Title: KR-HumanEval을 활용한 언어 모델의 한국어 프로그램 합성 성능 분석 (2024 HCLT)

## Benchmark Version
* data/v0/KR_HumanEval.jsonl: original benchmark
* data/v1/KR_HumanEval.jsonl: original benchmark with improved translation, taking into account the programming context.
  
## Acknowledgement
* [HumanEval](https://github.com/openai/human-eval)
