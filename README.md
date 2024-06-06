# llmbench
A library for validating and benchmarking LLMs inference.

## Run ScaleLLM Benchmark

```
python3 scalellm_run_benchmark.py --input_file /data/dataset/F_alpaca_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=16
```

## Run vllm Benchmark

```
python3 vllm_run_benchmark.py --input_file /data/dataset/Chatbot_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=16
```

## Run tensorrt_llm Benchmark

#### 1. Download TensorRT-LLM

```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

## 2. Huggingface Model Convert to TensorRT CKPT

```
python TensorRT-LLM/examples/qwen/convert_checkpoint.py --model_dir /data/qwen-7b --output_dir /data/qwen-7b-ckpt --dtype float16
```

重要参数：
- --workers 并行个数（tensor并行个数）
- --model_dir huggingface模型的位置
- --dtype 类型
- --output_dir 输出ckpt位置

## 3. Build TensorRT Engine

```
trtllm-build --checkpoint_dir /data/qwen-7b-ckpt --gemm_plugin float16 --use_gemm_plugin float16 --use_gpt_attention_plugin float16  --max_batch_size 256 --output_dir  /data/qwen-7b-engine
```

- --max_batch_size batch_size
- --max_input_len input length
- --max_output_len output length
- --output_dir output directory
- --checkpoint_dir ckpt directory
- --workers parallel number (tensor parallel number)

## 4. Run TensorRT-LLM on single GPU

```
python3 tensorrtllm_run_benchmark.py  --max_output_len=100  --tokenizer_dir /data/llama-2-7b-hf --engine_dir /data/llama-2-7b-engine --input_file /data/dataset/Chatbot_group_10_2.json --batch_size 16
```

## 5. Run TensorRT-LLM on two GPUs

```
mpirun -n 2 python run.py --max_output_len=100 --every_batch_cost_print True --tokenizer_dir /data/tensorrtllm_test/opt-13b/ --engine_dir /data/tensorrtllm_test/opt-13b-trtllm-build/ --input_file /data/opt-13b-test/Chatbot_group_10.json --batch_size 8
```
