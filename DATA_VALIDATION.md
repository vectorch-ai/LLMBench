## Reproduce

#### scalellm

python3 scalellm_run_data_validation.py --input_file /data/dataset/Chatbot_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=1

python3 scalellm_run_data_validation.py --input_file /data/dataset/F_alpaca_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=1

#### vllm

python3 vllm_run_data_validation.py --input_file /data/dataset/Chatbot_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=1

python3 vllm_run_data_validation.py --input_file /data/dataset/F_alpaca_group_10_2.json --model_dir=/data/llama-2-7b-hf --batch_size=1


## F_alpaca

#### case1
scalellm's output:
Prompt: 'I want you to act as an economist. Please answer the following question with no more than 50 words. Question: For a car, what scams can be plotted with 0% financing vs rebate?', Generated text: '\nI want you to act as an economist. Please answer the following question with no more than 50 words.\nFor a car, what scams can be plotted with 0% financing vs rebate?\nThe following question is for a car, what scams can be plotted with 0% financing vs rebate?\nThe following question is for a car, what scams can be plotted with 0% financing vs rebate?\n'

vllm's output:
Prompt: 'I want you to act as an economist. Please answer the following question with no more than 50 words. Question: For a car, what scams can be plotted with 0% financing vs rebate?', Generated text: '\nI want you to act as an economist. Please answer the following question with no more than 50 words.\nQuestion: For a car, what scams can be plotted with 0% financing vs rebate?\nhttps://essaysprompt.com/wp-content/uploads/2020/10/19-2.png 0 0 https://essaysprompt.com/wp-content/upload'


#### case2

scalellm's output:
Prompt: 'I want you to act as an economist. Please answer the following question with no more than 50 words. Question: Where should I be investing my money?', Generated text: '\nI want you to act as an economist. Please answer the following question with no more than 50 words.\nWhere should I be investing my money?\nI want you to act as an economist. Please answer the following question with no more than 50 words.\nWhere should I be investing my money?\nI want you to act as an economist. Please answer the following question with no more than 50 words.\nI want you to'

vllm's output:
Prompt: 'I want you to act as an economist. Please answer the following question with no more than 50 words. Question: Where should I be investing my money?', Generated text: '\nI want you to act as an economist. Please answer the following question with no more than 50 words.\nQuestion: Where should I be investing my money?\nI want you to act as an economist. Please answer the following question with no more than 50 words. Question: Where should I be investing my money? I want you to act as an economist. Please answer the following question with no more than 50 words. Question: Where'

##chatbot

#### case1

scalellm's output:
Prompt: "I don't want to tell the truth.", Generated text: "I don't want to tell the truth.\nI don't want to tell the truth\nI don't want to tell the truth, I don't want to tell the truth\nI don't want to tell the truth, I don't want to tell the truth, I don't want to tell the truth\nI don't want to tell the truth, I don't want to tell the truth, I don't want to tell the truth"

vllm's output:
Prompt: "I don't want to tell the truth.", Generated text: " I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth. I don't want to tell the truth."


#### case2

scalellm's output:
Prompt: "I don't care it was his fault for leaving the account un-locked and he should be grateful I didn't ever withdraw ever greater amount.", Generated text: "\n\nI don't care it was his fault for leaving the account un-locked and he should be grateful I didn't ever withdraw ever greater amount\nI don't care it was his fault for leaving the account un-locked and he should be grateful I didn't ever withdraw ever greater amount of money\nI don't care it was his fault for leaving the account un-locked and he should be grateful I didn't ever withdraw ever greater amount of money\nI"

vllm's output:
Prompt: "I don't care it was his fault for leaving the account un-locked and he should be grateful I didn't ever withdraw ever greater amount.", Generated text: "\nI don't care that he's a good friend and I'm a good friend.\nI don't care that he's a good friend and I'm a good friend and I'm a good friend and he's a good friend.\nI don't care that he's a good friend and I'm a good friend and I'm a good friend and he's a good friend and I'm a good friend.\nI"
