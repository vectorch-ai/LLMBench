# llmbench
A library for validating and benchmarking LLMs inference.

## Run ScaleLLM Benchmark

python3 scalellm_run_benchmark.py --input_file /data/dataset/F_alpaca_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=16

## Run vllm Benchmark

python3 vllm_run_benchmark.py --input_file /data/dataset/Chatbot_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=16

## Run tensorrt_llm Benchmark

#### Convert to trt ckpt

/TensorRT-LLM/examples/qwen# python3 convert_checkpoint.py --model_dir /data/qwen-7b --output_dir /data/qwen-7b-ckpt

#### Build trt engine

trtllm-build --checkpoint_dir /data/qwen-7b-ckpt --gemm_plugin float16 --max_batch_size 256 --output_dir  /data/qwen-7b-engine

#### Run tensorrt_llm

python3 tensorrtllm_run_benchmark.py  --max_output_len=100  --tokenizer_dir /data/llama-2-7b-hf --engine_dir /data/llama-2-7b-engine --input_file /data/dataset/Chatbot_group_10_2.json --batch_size 16

