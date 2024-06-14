# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorrt_llm
import torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from utils import (DEFAULT_HF_MODEL_DIRS, load_tokenizer, read_model_name,
                   throttle_generator)

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_format', type=str)
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behaviour'
    )
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument('--disable_output_print',
                        default=True,
                        help="Used to print all outputs")
    parser.add_argument('--every_batch_cost_print',
                        default=False,
                        help="User to print every batch inference cost.") 
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default='gpt2')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        help="The directory of LoRA weights")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")

    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")

    return parser.parse_args(args=args)

def read_json_input_v2(input_file=None):
    input_prompt = []
    if input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                instruct = data['instruction']
                prompt = data['input']
                input_prompt.append(instruct + ' ' + prompt)
    return input_prompt

def read_json_input(input_file=None):
    input_prompt = []
    if input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
            for idx, obj in enumerate(data):
                prompt = obj['prompt']
                input_prompt.append(prompt)
    return input_prompt           

def decode_prompt(batch_input_prompt,
                  tokenizer,
                  add_special_tokens = True,
                  max_input_length = 923):
    batch_input_ids = []
    for prompt in batch_input_prompt:
        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens = add_special_tokens,
            truncation = True,
            max_length=max_input_length)
        batch_input_ids.append(input_ids)
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids

def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text,
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.read()
                input_ids = tokenizer.encode(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)
                batch_input_ids.append(input_ids)
        elif input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
                for idx, obj in enumerate(data):
                    prompt = obj['prompt']
                    input_ids = tokenizer.encode(
                        prompt,
                        add_special_tokens = add_special_tokens,
                        truncation=True,
                        max_length=max_input_length)
                    batch_input_ids.append(input_ids)
        else:
            print('Input file format not supported.')
            raise SystemExit
    print("num_prepend_vtokens:")
    print(num_prepend_vtokens)
    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]
    if model_name == 'glm_10b':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 output_logits_npy=None):
    batch_size, num_beams, _ = output_ids.size()
    if output_csv is None and output_npy is None:
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                print(
                    f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)

    if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
        input_lengths = torch.Tensor(input_lengths)
        context_logits = torch.cat(context_logits, axis=0)
        generation_logits = generation_logits.squeeze(1)
        last_token_ids = torch.cumsum(input_lengths, dim=0).int().cuda()
        batch_size = input_lengths.size(0)

        vocab_size_padded = context_logits.shape[-1]
        context_logits = context_logits.reshape([1, -1, vocab_size_padded])
        contex_last_token_logits = torch.index_select(context_logits, 1,
                                                      last_token_ids - 1).view(
                                                          batch_size, 1,
                                                          vocab_size_padded)
        generation_logits = torch.cat(
            [contex_last_token_logits, generation_logits], axis=1)
        generation_logits = generation_logits.reshape(
            -1, num_beams, generation_logits.shape[1], generation_logits.
            shape[2])  # [batchSize, beamWidth, maxOutputLength, vocabSize]
        # Save context logits
        output_context_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_context"
        output_context_logits_file = Path(output_context_logits_npy)
        context_outputs = np.array(
            context_logits.squeeze(0).cpu().contiguous(),
            dtype='float32')  # [promptLengthSum, vocabSize]
        np.save(output_context_logits_file, context_outputs)
        # Save generation logits
        output_generation_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_generation"
        output_generation_logits_file = Path(output_generation_logits_npy)
        generation_outputs = np.array(generation_logits.cpu().contiguous(),
                                      dtype='float32')
        np.save(output_generation_logits_file, generation_outputs)


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name = read_model_name(args.engine_dir)
    if args.tokenizer_dir is None:
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        tokenizer_type=args.tokenizer_type,
    )

    # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
    # stop_words_list = [[" London"], ["eventually became"]]
    # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
    # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
    stop_words_list = None

    # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
    # bad_words_list = [[" chef"], [" eventually, chef before"]]
    # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
    # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
    bad_words_list = None

    # prompt_template = None
    # if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
    #     prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

    if args.data_format == 'v2':    
        input_prompt = read_json_input_v2(input_file=args.input_file)
    else:
        input_prompt = read_json_input(input_file=args.input_file)
    
    '''
    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=args.input_text,
                                  prompt_template=prompt_template,
                                  input_file=args.input_file,
                                  add_special_tokens=args.add_special_tokens,
                                  max_input_length=args.max_input_length,
                                  pad_id=pad_id,
                                  num_prepend_vtokens=args.num_prepend_vtokens,
                                  model_name=model_name)
    input_lengths = [x.size(0) for x in batch_input_ids]
    '''

    print("=================Debug Settings==================")
    if args.every_batch_cost_print is False:
        print('every batch cost print is disable, you can enable by --every_batch_cost_print True')

    if args.disable_output_print is True:
        print("print output is disabled, you can enable printing output by --disable_output_print False")
    print("=================================================")

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(engine_dir=args.engine_dir,
                         lora_dir=args.lora_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode,
                         lora_ckpt_source=args.lora_ckpt_source,
                         )
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=args.batch_size,
            max_input_len=args.max_input_length,
            max_output_len=args.max_output_len,
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
        )
    runner = runner_cls.from_dir(**runner_kwargs)

    with torch.no_grad():
        loop_count = math.ceil(len(input_prompt) / args.batch_size)
        outputs = []
        input_lengths = []
        total_time_cost = 0
        print("================Inputs & BatchSize===============")
        print("total_input_count:" + str(len(input_prompt)))
        print("loop_count:" + str(loop_count))
        print("batch_size:" + str(args.batch_size))
        print("=================================================")
        for i in range(0, loop_count):
            t1 = time.time()
            batch_input_prompt = input_prompt[i*args.batch_size:(i+1)*args.batch_size]
            batch_input_ids = decode_prompt(batch_input_prompt, tokenizer, args.add_special_tokens, args.max_input_length)
            batch_input_lengths = [x.size(0) for x in batch_input_ids]
            input_lengths.append(batch_input_lengths)

            batch_outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=args.max_output_len,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                lora_uids=args.lora_task_uids,
                prompt_table_path=args.prompt_table_path,
                prompt_tasks=args.prompt_tasks,
                streaming=args.streaming,
                output_sequence_lengths=True,
                return_dict=True)
            torch.cuda.synchronize()
            t2 = time.time()
            if args.every_batch_cost_print:
                print('one batch time cost: %ss:' % ((t2 - t1)))
            total_time_cost += t2-t1
            outputs.append(batch_outputs)
        print("=================Time Consumption================")
        print("total time cost: %ss:" % total_time_cost)
        print("average time cost: %ss:" % (total_time_cost / loop_count))
        print("=================================================")

    if args.streaming:
        for curr_outputs in throttle_generator(outputs,
                                               args.streaming_interval):
            if runtime_rank == 0:
                output_ids = curr_outputs['output_ids']
                sequence_lengths = curr_outputs['sequence_lengths']
                print_output(tokenizer,
                             output_ids,
                             input_lengths,
                             sequence_lengths,
                             output_csv=args.output_csv,
                             output_npy=args.output_npy)
    else:
        if args.disable_output_print is True:
            return
        if runtime_rank == 0:
            for i in range(0, loop_count):
                batch_outputs = outputs[i]
                batch_input_lengths = input_lengths[i]
                print("===============batch_size(" + str(len(batch_input_lengths)) + ")==================")
                output_ids = batch_outputs['output_ids']
                sequence_lengths = batch_outputs['sequence_lengths']
                context_logits = None
                generation_logits = None
                # cum_log_probs = None
                # log_probs = None
                if runner.gather_context_logits:
                    context_logits = outputs['context_logits']
                if runner.gather_generation_logits:
                    generation_logits = outputs['generation_logits']
                print_output(tokenizer,
                             output_ids,
                             batch_input_lengths,
                             sequence_lengths,
                             output_csv=args.output_csv,
                             output_npy=args.output_npy,
                             context_logits=context_logits,
                             generation_logits=generation_logits,
                             output_logits_npy=args.output_logits_npy)
                print("=============================================")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
