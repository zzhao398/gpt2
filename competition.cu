//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

// Self-contained competition benchmark runner.
// Usage: ./competition_metrics [output_file]
// All benchmark data is embedded - no external benchmark files needed.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils/utils.h"
#include "utils/tokenizer.h"
#include "utils/cuda_utils.cuh"

#include "gpt2.cuh"

const char *benchmark_results_filename = "competition_results.txt";
const char *generation_output_filename = "generation_outputs.txt";
const char *checkpoint_file = "/work/hdd/bche/Project_GPT/gpt2_124M.bin";

#define TEMPERATURE 0.7  /* Should be > 0 */

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void get_substring_up_to_space(char *input, char *output) {
    char *space_pos = strchr(input + 1, ' ');

    if (space_pos != NULL) {
        size_t length = space_pos - input;
        strncpy(output, input, length);
        // null-terminate output
        output[length] = '\0';
    } else {
        // if no space, copy the entire string
        strcpy(output, input);
    }
}

int* tokenizer_encode(Tokenizer *tokenizer, const char *input_str, int *num_tokens) {
    if (tokenizer->init_ok == 0) {
        *num_tokens = 0;
        return NULL;
    }

    // arbitrary initial number of tokens
    int max_tokens = 128;
    int *tokens = (int*)mallocCheck(max_tokens * sizeof(int));
    int token_count = 0;

    char temp_str[strlen(input_str) + 1];
    strcpy(temp_str, input_str);

    // pointer to keep track of the remaining string to process
    char *remaining_input_str = temp_str;
    while (*remaining_input_str != '\0') {
        char to_search[100];
        get_substring_up_to_space(remaining_input_str, to_search);
        size_t token_len = strlen(to_search);

        // try to find longest matching token from the token_table (not perfect, but probably good enough heuristic)
        int best_token = -1;
        int best_token_length = -1;
        for (int i = 0; i < tokenizer->vocab_size; ++i) {
            const char *token = tokenizer->token_table[i];
            int tokenlen = strlen(token);

            if (strncmp(to_search, token, tokenlen) == 0) {
                if (tokenlen > best_token_length) {
                    best_token_length = tokenlen;
                    best_token = i;
                }
            }
        }
        assert(best_token != 0);
        tokens[token_count++] = best_token;
        // move the pointer ahead by the length of the token
        remaining_input_str += best_token_length;

        // if reached current tokens array capacity, resize
        if (token_count >= max_tokens) {
            // double the capacity
            max_tokens *= 2;
            tokens = (int*)realloc(tokens, max_tokens * sizeof(int));
            if (!tokens) {
                fprintf(stderr, "next_token_gen - Memory allocation failed during token array resizing.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    *num_tokens = token_count;
    return tokens;
}

// TODO: Batch for performance.
int sample_next_token_temperature(const float* logits, int vocab_size, float temperature) {
    // find max logit, to be used for numerical stability
    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float norm = 0.0f;
    float* probabilities = (float*)mallocCheck(vocab_size * sizeof(float));

    // compute probabilities using softmax with numerical stability and temperature
    for (int i = 0; i < vocab_size; i++) {
        probabilities[i] = expf((logits[i] - max_logit) / temperature);
        norm += probabilities[i];
    }

    // normalize
    for (int i = 0; i < vocab_size; i++) {
        probabilities[i] /= norm;
    }

    // sample from distribution
    float random_value = (float)rand() / RAND_MAX;
    float cumulative = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        cumulative += probabilities[i];
        if (random_value < cumulative) {
            free(probabilities);
            return i;
        }
    }

    free(probabilities);

    // default fallback (last token in vocab)
    return vocab_size - 1;
}

void append_to_file(const char *filename, const char *string_to_append) {
    FILE *file = fopenCheck(filename, "a");
    if (file == NULL) {
        perror("next_token_gen - Error opening file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%s", string_to_append);
    fclose(file);
}

void log_generation_output(const char* input_sequence, const char* output_text) {
    append_to_file(generation_output_filename, "next_token_gen - INPUT TEXT:\n");
    append_to_file(generation_output_filename, input_sequence);
    append_to_file(generation_output_filename, "\n\nGENERATED TEXT:\n");
    append_to_file(generation_output_filename, output_text);
    append_to_file(generation_output_filename, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
}

class Benchmark {
  public:
    Benchmark (const std::string name, int batches, int max_gen_len) 
        : name_(name), batches_(batches), max_gen_len_(max_gen_len) {}

    void start () { start_ = std::chrono::steady_clock::now(); }
    void end () { end_ = std::chrono::steady_clock::now(); }

    double runtime () const {
        return std::chrono::duration<double>(end_ - start_).count();
    }

    double tokens_per_second () const {
        return static_cast<double>(max_gen_len_ * batches_) / static_cast<double>(runtime());
    }

    std::string to_str () const {
        return name_ + "," + std::to_string(runtime()) + "," + std::to_string(tokens_per_second()) + "\n";
    }

    const std::string& name() const { return name_; }

  private:
    std::string name_;
    int batches_;
    int max_gen_len_;

    std::chrono::time_point<std::chrono::steady_clock> start_;
    std::chrono::time_point<std::chrono::steady_clock> end_;
};

void log_benchmark (const std::string benchmark_results_filename, const Benchmark& benchmark) {
    append_to_file(benchmark_results_filename.c_str(), benchmark.to_str().c_str());
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void gpt2_generate_next_token_batched(GPT2* model, int batches, int input_stride, int* input_tokens, int input_length, int* output_tokens) {
    // 0. Assert assumptions
    assert(input_length <= model->config.max_seq_len);
    
    // 1. Copy data
	int padded_input_length = (input_length + 3) / 4 * 4;
	int * gpt_input = (int *)mallocCheck(batches * padded_input_length * sizeof(int));

    for (int batch_id = 0; batch_id < batches; batch_id++) {
        memcpy(
            &gpt_input[batch_id * padded_input_length],
            &input_tokens[batch_id * input_stride],
            input_length * sizeof(int)
        );
    }

	if (padded_input_length - input_length > 0) {
        for (int batch_id = 0; batch_id < batches; batch_id++) {
            memset(&gpt_input[batch_id * padded_input_length + input_length], 0, (padded_input_length - input_length) * sizeof(int));
        }
    }

    // 2. Run forward
    gpt2_forward(model, gpt_input, batches, padded_input_length);

    // 3. Copy and sample logits
    for (int batch_id = 0; batch_id < batches; batch_id++) {
        float* logits_gpu = model->acts.output +
            (batch_id * padded_input_length + (input_length - 1)) *
            model->config.padded_vocab_size;
        float* logits_cpu = (float*)mallocCheck(model->config.vocab_size * sizeof(float));

        // move logits back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
        cudaCheck(cudaMemcpy(logits_cpu, logits_gpu, model->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));


        output_tokens[batch_id] = sample_next_token_temperature(logits_cpu, model->config.vocab_size, TEMPERATURE);

        free (logits_cpu);
    }
}

void gpt2_generate_text_batched(GPT2* model, const std::vector<std::string>& inputs, int batches, int max_gen_length, std::vector<char*>& outputs) {
    // 1) Tokenize inputs.
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "/work/hdd/bche/Project_GPT/gpt2_tokenizer.bin");

    int num_input_tokens = -1;
    std::vector<int*> input_tokens(batches);

    for (int i = 0; i < inputs.size(); i++) {
        int cur_input_tokens = 0;
        input_tokens[i] = tokenizer_encode(&tokenizer, inputs[i].c_str(), &cur_input_tokens);

        // Verify batch has equal lengths!
        if (cur_input_tokens != num_input_tokens && i != 0) {
            std::cerr << "Token count mismatch: " << num_input_tokens << ", " << cur_input_tokens << std::endl;
            std::exit(1);
        }
        num_input_tokens = cur_input_tokens;
    }

    std::cout << "Benchmark Run [ Input Tokens: " << num_input_tokens 
                                << ", Batches: " << batches 
                                << ", Generation Length: " << max_gen_length << " ]" << std::endl;

    // 2) Allocate output buffers.
    int  single_batch_size   = num_input_tokens + max_gen_length;
    int* generated_sequences = (int*)mallocCheck(batches * single_batch_size * sizeof(int));

    for (int i = 0; i < batches; i++) {
        memcpy(&generated_sequences[i * single_batch_size], input_tokens[i], num_input_tokens * sizeof(int));
    }

    // 3) Generate next tokens.
    int seq_len_itr = num_input_tokens;
    int* next_tokens = (int*)mallocCheck(sizeof(int) * batches); // TODO: Convert all malloc/free to new/delete !!!
    for (int i = 0; i < max_gen_length; ++i) {
        gpt2_generate_next_token_batched(
            model, 
            batches, 
            single_batch_size,
            generated_sequences, 
            seq_len_itr, 
            next_tokens
        );

        for (int batch_id = 0; batch_id < batches; batch_id++) {
            if (seq_len_itr < model->config.max_seq_len) {
                generated_sequences[batch_id * single_batch_size + seq_len_itr] = next_tokens[batch_id];
            }
        }
        seq_len_itr += 1;
    }
    free(next_tokens);

    // 4) Detokenize outputs.
    for (int batch_id = 0; batch_id < batches; batch_id++) {
        outputs.push_back((char*)mallocCheck(sizeof(char) * 4096));
        memset(outputs.back(), 0, sizeof(char) * 4096);

        for (int i = num_input_tokens; i < num_input_tokens + max_gen_length; ++i) {
            const char* next_token_str = tokenizer_decode(&tokenizer, generated_sequences[batch_id * single_batch_size + i]);
            strcat(outputs[batch_id], next_token_str);
        }
    }

    // 5) Cleanup.
    for (int i = 0; i < input_tokens.size(); i++) { free(input_tokens[i]); }

    free(generated_sequences);
    tokenizer_free(&tokenizer);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EMBEDDED BENCHMARK DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

struct EmbeddedBenchmark {
    const char* name;
    int batches;
    int max_gen_len;
    const char* prompt;  // Each batch uses the same prompt
};

static const char* PROMPT_SHORT = "the meaning of life is ";

static const char* PROMPT_MEDIUM =
    "Course Assistant A: CUDA or compute unified device architecture is super cool. "
    "With all of the AI hype, taking ECE 408 will be great for your understanding of "
    "how these architectures are possible. Whether you have a CS or EE background, "
    "I'm sure you can find applications of your learning to this course. Student:";

static const char* PROMPT_LONG =
    "In the quiet hours before sunrise, the city felt like a sleeping organism, "
    "its lights blinking like distant neurons. A soft fog rolled over the streets, "
    "blurring the edges of everything familiar. Somewhere, a train announced its "
    "departure with a low, resonant hum, echoing through the steel and glass canyons. "
    "The sound reminded her of possibility\xe2\x80\x94the kind that exists only when the world "
    "hasn't yet woken. She walked without purpose, tracing the rhythm of her own breath, "
    "letting each step dissolve into the silence. For a moment, she imagined that if she "
    "listened closely enough, the city might whisper back.";

static const EmbeddedBenchmark BENCHMARKS[] = {
    // Warmup
    {"warmup",          1,   50, PROMPT_SHORT},
    // 1-batch sweeps
    {"1b_50g_short",    1,   50, PROMPT_SHORT},
    {"1b_50g_medium",   1,   50, PROMPT_MEDIUM},
    {"1b_50g_long",     1,   50, PROMPT_LONG},
    {"1b_100g_short",   1,  100, PROMPT_SHORT},
    {"1b_100g_medium",  1,  100, PROMPT_MEDIUM},
    {"1b_100g_long",    1,  100, PROMPT_LONG},
    {"1b_200g_long",    1,  200, PROMPT_LONG},
    {"1b_500g_long",    1,  500, PROMPT_LONG},
    // 4-batch sweeps
    {"4b_50g_short",    4,   50, PROMPT_SHORT},
    {"4b_50g_medium",   4,   50, PROMPT_MEDIUM},
    {"4b_50g_long",     4,   50, PROMPT_LONG},
    {"4b_200g_long",    4,  200, PROMPT_LONG},
    // Large batches
    {"8b_200g_long",    8,  200, PROMPT_LONG},
    {"16b_200g_long",  16,  200, PROMPT_LONG},
    {"32b_200g_long",  32,  200, PROMPT_LONG},
};

static const int NUM_BENCHMARKS = sizeof(BENCHMARKS) / sizeof(BENCHMARKS[0]);

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

int main(int argc, char** argv) {
    std::string benchmark_filename = "competition_results.txt";
    if (argc >= 2) {
        benchmark_filename = argv[1];
    }

    // 1. Prelude (seed random, cuBLAS setup, TF32, etc.).
    srand(time(NULL));
    
    cublasCheck(cublasCreate(&cublas_handle));
    int enable_tf32 = 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // 2. Load GPT2
    GPT2 model;
    gpt2_build_from_checkpoint(&model, checkpoint_file);
    
    // 3. Write Header
    std::string benchmark_header = "name,runtime(s),tokens/second\n";
    append_to_file(benchmark_filename.c_str(), benchmark_header.c_str());

    // 4. Run embedded benchmarks.
    std::vector<Benchmark> benchmarks;
    for (int b = 0; b < NUM_BENCHMARKS; b++) {
        const EmbeddedBenchmark& eb = BENCHMARKS[b];

        std::vector<std::string> inputs;
        for (int i = 0; i < eb.batches; i++) {
            inputs.push_back(std::string(eb.prompt));
        }

        Benchmark cur(eb.name, eb.batches, eb.max_gen_len);
        std::vector<char*> outputs;

        cur.start();
        gpt2_generate_text_batched(&model, inputs, eb.batches, eb.max_gen_len, outputs);
        cur.end();

        for (int i = 0; i < eb.batches; i++) {
            log_generation_output(inputs[i].c_str(), outputs[i]);
            free(outputs[i]);
        }

        benchmarks.push_back(cur);
        log_benchmark(benchmark_filename, cur);
    }

    // 5. Compute geometric mean of tokens/second (excluding warmup), matching competition scoring.
    double log_sum = 0.0;
    int scored_count = 0;
    for (const auto& b : benchmarks) {
        if (b.name() == "warmup") continue;
        log_sum += log(b.tokens_per_second());
        scored_count++;
    }

    if (scored_count > 0) {
        double geomean = exp(log_sum / scored_count);
        printf("\n========================================\n");
        printf("  COMPETITION SCORE (geomean tok/s): %.2f\n", geomean);
        printf("========================================\n");

        std::string score_line = "GEOMEAN," + std::to_string(0.0) + "," + std::to_string(geomean) + "\n";
        append_to_file(benchmark_filename.c_str(), score_line.c_str());
    }

    return EXIT_SUCCESS;
}
