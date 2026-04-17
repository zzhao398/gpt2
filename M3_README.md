# ECE408 Final Project - GPT-2

## Milestone 3

## Table of Contents

* [Introduction](#introduction)
* [Milestone Requirements](#milestone-requirements)
* [Implementation and Testing](#implementation-and-testing)
* [Performance Analysis and Profiling ](#performance-analysis-and-profiling)
* [Final Report](#final-report)
* [Deliverables](#deliverables)
* [Submission and Grading](#submission-and-grading)
* [Additional Details/Hints for Optimizations](#additional-detailshints-for-optimizations)
    + [Required Optimizations Details](#required-optimizations-details)
    + [Additional Optimizations Details](#additional-optimizations-details)

## Introduction

This milestone focuses on further optimizing the GPU kernels from Milestone 2. Your aim is to create the world's best possible implementation of the GPT-2 model's forward pass kernels on the GPU.

This is where you let your imaginations fly and CUDA/ML knowledge shine! 

You will also profile your optimizations and present your findings in a final report, one detailed enough for you to remember your work forever!

## Milestone Requirements

This milestone is worth a total of 50% of your project grade. The coding portion is split into two mandatory categories: **Required Optimizations (20%)** and **Additional Optimizations (10%)**.

For full credit on this milestone, you will need to:
- Complete the below list of required optimizations
- Select and implement a subset of the **Additional Optimizations** such that their combined point values equal at least 10%.
- Profiling:
  - Find each individual required and additional optimization's effectiveness through various profiling metrics
  - Determine **specific reasons** for speedups/slowdowns/no effect
  - Describe how different optimizations work together (or against each other)
  - Compare the final optimized implementation with baseline kernels from Milestone 1
- Present findings:
  - Final Report

### Required optimizations (20% Total):

You **must** implement all of the following optimizations:

| Optimization | Description                                  | Weight |
| ------------ | -----------                                  | ------ |
| req_4        | Flash Attention                              | 6%     |
| req_5        | Local/Windowed Attention                     | 2%     |
| req_6        | KV-Caching                                   | 12%    |

### Additional optimizations (10%+ Total):

To earn the remaining 10% of your coding grade, you must mix and match from the pool of additional optimizations below. This list includes standard advanced optimizations as well as excellent ideas proposed by teams during Milestone 2.

You may pick any combination of the following, so long as the weights add up to at least 10%. *(Note: If your selected optimizations exceed 10%, the extra points may be considered for extra credit, up to 10% max when combined with leaderboard extra credit).*

| Optimization | Description                                  | Weight |
| ------------ | -----------                                  | ------ |
| op_7         | Configuration Sweep/Optimization             | 3%     |
| op_8         | Constant Memory                              | 2%     |
| op_9         | `__restrict__`                               | 1%     |
| op_10        | Split-K                                      | 3%     |
| op_11        | Data Swizzling                               | 3%     |
| op_12        | Shared Memory Padding                        | 3%     |
| op_13        | Host Allocation Data Alignment               | 2%     |
| op_14        | Basic CUTLASS Device GEMM                    | 3%     |
| op_15        | Advanced CUTLASS with CuTe                   | 5%     |
| op_16        | Advanced CUTLASS with Fused Epilogues        | 4%     |

**CUTLASS Mutually Exclusive Rule**: You may only choose **ONE** of the CUTLASS optimizations (`op_14`, `op_15`, or `op_16`) to implement for credit. You cannot stack these for combined points. Also, you may not use CUTLASS or any other NVIDIA libraries to implement the other optimizations for you.

**Important**: Please see the additional details and hints provided at the end of this document before beginning your implementations.

## Implementation and Testing

For this milestone, most optimizations will be graded on accuracy/correctness using the existing verification scripts (`test_gpt2_kernels.cu` or `test_gpt2.cu`).

However, because you are implementing structural architectural changes (like `req_5: Local/Windowed Attention`) and selecting your own mix of additional optimizations, your group is ultimately responsible for demonstrating the correctness of any optimizations that intentionally cause deviations from our provided op-level and end-to-end solution values.

In other words, if a chosen optimization changes the math such that it fails tests in either `test_gpt2_kernels.cu` or `test_gpt2.cu`, you will need to develop a sufficient verification method (for example, using the *Perplexity* scoring system to evaluate the model's language modeling accuracy) and justify in your final report why the optimization is implemented correctly. Without this justification, you **may NOT** receive full credit. For optimizations that do not cause deviations, the existing verification scripts are sufficient.

Since `req_5` (Local/Windowed Attention) changes the attention mechanism from full attention to local/windowed attention, it is expected that this optimization will cause deviations from our provided solution values. See the additional details section for more information on how to verify this specific optimization.

## Performance Analysis and Profiling 

Refer to [M2_README.md](M2_README.md) for profiling information. 

## Final Report

After performing both system-level profiling and kernel-level profiling, your team will need to submit a final report detailing the following:
- **Implementation Details**
    - For each optimization (both required and your chosen additional ones), describe how you implemented it.
    - If your chosen additional optimizations did not yield performance improvements, explain why you think that is the case.
    - Explain how you verified each implementation's correctness (especially if the final model outputs deviated from the baseline).
    - Discuss any challenges you faced during implementation and how you overcame them.


- **Performance Analysis of Milestone 3 Optimizations**
    - Compare the performance of your final optimized kernels with the baseline implementations from Milestone 1.
    - Be sure to cover **all** new Milestone 3 optimizations (required and additional).
    - Analyze why each optimization was effective (or why it wasn't).
    - What bottlenecks are addressed by your final version of each kernel?
    - Logical conclusions should be drawn in addition to just reporting metrics.
    - Detailed metrics and graphs can be used to support your analysis. 
    - End-to-end performance comparisons should also be included here.

- **Pretty Pictures and Data**
    - We'd really appreciate some nice charts or graphs of performance metrics! (annotate them and use them to supplement your analysis instead of a raw screenshot would be even better)
    - Comparison tables for things like execution time can be helpful too. 

- **Interesting Findings**
    - Anything you find interesting and/or surprising from the profiling results.

When writing this report, you should make full use of the detailed metrics provided by Nsight-Systems and Nsight-Compute. 

You should be detailed and in-depth about your findings and comparisons from profiling, **do NOT** just copy some metrics from Nsight, explain what those metrics are, and call it a day! Make sure to explain and provide insights on **why** the metrics look the way they do, and what implementations decisions led to those metrics.

This report does not have a length requirement. A long report packed with different metrics and surface-level analysis is **not** what we're looking for (*that's the job of the Nsight tools*). Your job is to interpret the data and provide **concise yet insightful conclusions**. This means we prefer quality over quantity, and some good insights on only a couple kernels is better than a bunch of surface-level analysis/boring metrics on every kernel.

Your report should be a document that records all high level design decisions, optimizations, and performance analysis done throughout the entire GPT Final Project. The report should be written in a way that is accessible to a general audience, but also contains details that would allow you (or anyone else) to reproduce your results if you look back to this project 5 years later.

## Deliverables

| Step | Deliverables                                     |
| ---- | ------------------------------------------       |
| 1    | Implemented M3 Required Optimizations            |
| 2    | Implemented M3 Additional Optimizations          |
| 3    | Profiling and Performance Analysis               |
| 4    | Milestone 3 Demo                                 |
| 5    | Final Report                                     |

## Submission and Grading

To submit your final report, push it as a **PDF file** to the root directory of your GitHub main branch named `[team_name]_Final_Report.pdf`, where `[team_name]` is your team's name. There is no strict format requirement for them as it will be manually graded, but make sure the file is typed (not hand-written) and correctly named.

### Code Submission

Your code should be submitted in the main branch of your repository. To ensure the autograder can correctly interface with your mix-and-match selections, you must organize your code into specific folders.

Required optimizations must be placed in folders named `kernels_req_x`. Your chosen additional optimizations must be placed in folders named `kernels_op_x`.

For example, if your team chose to complete Configuration Sweeping (`op_7`), Constant Memory (`op_8`), Split-K (`op_10`), and Host Allocation Data Alignment (`op_13`) to reach your 10% requirement (with 1% extra credit), your directory structure should look exactly like this:
```
 sp26_ece408_[team_name]
  ├── kernels
  │   ├── attention.cuh
  │   ├── encoder.cuh
  │   ├── gelu.cuh
  │   ├── matmul.cuh
  │   └── ...            [unmodified kernels from M1]
  ├── kernels_req_4
  │   └── attention.cuh  [with Flash Attention]
  ├── kernels_req_5
  │   └── attention.cuh  [with Local/Windowed Attention]
  ├── kernels_req_6
  │   └── ...            [with KV-Caching]
  ├── kernels_op_7
  │   └── [configuration sweep script]
  ├── kernels_op_8
  │   └── ...            [with constant memory]
  ├── kernels_op_10
  │   └── ...            [with Split-K]
  ├── kernels_op_13
  │   └── gpt2.cuh       [modified to have aligned allocations]
  ├── gpt2.cuh
  ├── Makefile
  ├── README.md
  └── ...
```

**Important**: Any incorrect/non-functioning implementations will receive 0 correctness points. However, you may still earn points for your understanding and analysis of the optimization in your final report even if the implementation is incorrect/non-functioning.


## Additional Details/Hints for Optimizations

### Required Optimizations Details

**Note:** Lecture 23 goes through all of the required optimizations for this project in great detail. We highly recommend reviewing this lecture before starting your implementations!

#### req_4: Flash Attention

The FlashAttention series of papers proposed a monumental IO-aware attention optimization that is now widely used in the industry. The algorithm enables tiling of the attention matrix to address the memory bandwidth bottleneck in attention. You may choose to implement either the basic FlashAttention algorithm, outlined in the original [FlashAttention paper](https://arxiv.org/pdf/2205.14135), **or** the improved version, outlined in the [FlashAttention2 paper](https://tridao.me/publications/flash2/flash2.pdf). 

**Note:** Both FlashAttention and FlashAttention2 are long papers that may take a while to read fully, so we recommend focusing on the following sections:
  - If you choose to implement the basic FlashAttention: Introduction, Section 2.2 (Standard Attention Implementation), Section 3.1 (**Skip** discussions of Recomputation)
  - If you choose to implement FlashAttention2: Introduction, Section 2.2 (Standard Attention Implementation), Section 2.3.1 (Basic FlashAttention Algorithm), Section 3.1.1 (FlashAttention2 Algorithm). 
  
    *If you are using the FlashAttention2 paper on arxiv.org, there may be a minor mistake in the pseudo code outlined in section 3.1.1. Please use the paper link provided above to ensure you have the correct version.*

The papers have already provided pseudo code for the algorithms, your job is to translate them into efficient CUDA code.

#### req_5: Local/Windowed Attention

We have provided a reference CPU implementation in `cpu_kernels/local_attention.cuh` for this optimization. Make sure you understand what exactly local/windowed attention is based on the provided CPU code before starting this optimization. 
Additionally, we have provided a basic skeleton test script (*which you need to modify yourself*) for this optimization (`local_attn_verify.cu` and `local_attn_verify.slurm`). You may refer to these files when implementing and verifying your local/windowed attention optimization. 
**Important:** For grading and verification purposes, you need to submit using a **window size of 128** (as shown in `local_attention.cuh`). However, once you have verified your implementation with a window size of 128 using our provided test data, feel free to experiment with different window sizes and include your findings in the final report.
**Note:** Make sure to use `make local_attn_verify` to compile your code base instead of simply using `make` to ensure your modifications to the local attention verification script is compiled correctly.

#### req_6: KV-Caching

To understand KV-Caching, you first need to understand the bottleneck of autoregressive text generation. When a Large Language Model generates text, it does so one token at a time. To generate the 100th token, the model must compute the Attention scores for the new token against the previous 99 tokens. In a naive implementation, the model recalculates the Key (K) and Value (V) tensors for all 99 previous tokens from scratch every single time it generates a new word. 

KV-Caching eliminates this massive redundant computation. Instead of throwing away the Key and Value matrices after a token is generated, you will implement a system to persistently store ("cache") them in GPU memory. 

During the generation phase, your code will only compute the K and V for the single newest token, append it to the cache, and perform the attention math using the newly updated history. You will need to carefully manage memory allocations to hold this growing cache across multiple iterations of the forward pass without crashing or overwriting data.

### Additional Optimizations Details

#### op_7: Configuration Sweep/Optimization

For this optimization, create a script (python, bash, or any language of your choice) to systematically sweep through different configurations like block sizes, thread counts, loop unrolling, etc. 
*Note for loop unrolling: Loop unrolling can technically be applied to any loop in your code, but you should understand where you should focus on and why. Make sure to perform thorough profiling after implementing this optimization and be able to directly point out the effects of loop unrolling.*

Make sure you at least provide why and how you are sweeping each configuration, what you expected before the sweep, and what you found after the sweep. You should try a sufficient amount of configurations to ensure you are getting the best performance from this optimization. 

In short, you should be able to answer the following questions:
- How does your script sweep through different configurations?
- What configurations did you try?
- Why did you choose to focus on these specific parameters?
- What are your findings? Include expectations before and analysis/conclusion after.

**Make sure to submit your sweeping script by pushing it to the main branch of your group's repository.** Note that we will not grade this script. However, we will manually check that you have submitted a working sweeping script in the `kernels_op_7` folder. You should also refer this script in your final report's when discussion this optimization.

Note that there is not a minimum number of configurations you need to try, but make sure you try enough configurations to ensure you are getting the best performance from your kernels, and sufficient justifications of your choices and analysis in your final report will earn you full points for this optimization.

#### op_8: Constant Memory

Think about what data you can/should store in constant memory. Review the lecture covering constant memory if needed. Make sure to perform thorough profiling after implementing this optimization to see how it is actually improving performance (*or explain why it's not*). 

You may choose which kernel(s) to apply this optimization to, and providing sufficient justifications of your choices and analysis in your final report will earn you full points for this optimization.

#### op_9: `__restrict__`

Make sure you understand what `__restrict__` does and how it can help improve performance before starting. You should also be able to explain why (and where) you are using `__restrict__` in your code.

You may choose which kernel(s) to apply this optimization to, and providing sufficient justifications of your choices and analysis in your final report will earn you full points for this optimization.

#### op_10: Split-K

Split-K is a technique that splits the K dimension of a GEMM operation into smaller chunks, allowing for better parallelism and memory access patterns. This can lead to improved performance, especially for certain matrix sizes. For more details on Split-K, you may refer to the advanced optimizations lecture (Lecture 16). 

You may choose which kernel(s) to apply this optimization to. Correct implementation of this optimization to **at least 1 kernel** will earn you full points for this optimization, but you **need** to perform thorough profiling and provide detailed analysis in your final report on how this optimization is affecting performance and justify the implementation choices you made.

#### op_11: Data Swizzling (Shared Memory)

NVIDIA GPUs divide shared memory into 32 distinct "banks." You can think of these banks like checkout lanes at a grocery store. If 32 threads in a warp try to access data from 32 different lanes, the transactions happen instantly in parallel. But if multiple threads try to read from the *same* lane (a "bank conflict"), the hardware forces them to wait in line, serializing the requests and tanking your memory bandwidth. This often happens when reading data column-by-column from a standard memory tile.

"Swizzling" is a compute-based trick to fix this without changing the size of your array. By using a reordering operation, such as a bitwise XOR operation (`^`) between the row and column indices when calculating your shared memory addresses, you dynamically scramble (or "swizzle") the data layout. 

This mathematical permutation ensures that elements of the same column are no longer stacked directly on top of each other in the same memory bank. When threads go to read that column, they apply the exact same XOR math to find the data, pulling from 32 distinct banks in perfect parallel. 

#### op_12: Shared Memory Padding

This optimization solves the exact same bank conflict problem described in Data Swizzling, but it fixes it using the memory footprint instead of mathematics.

If the inner dimension of your 2D shared memory tile aligns perfectly with the number of hardware memory banks, any thread warp reading a column will trigger a massive bank conflict. Padding solves this by deliberately altering the memory stride. By adding "dummy" padding to the inner dimension of your shared memory allocation, you force consecutive elements of a column to shift into different memory banks.

There is no single "correct" way to pad. Some implementations add a minimal amount of padding to slightly stagger the rows, while others significantly increase (or even double) the allocation footprint to drastically spread out the bank accesses. You are expected to explore different padding dimensions. 

Your goal is to use your profiling tools to find a strategy that successfully eliminates bank conflicts without wasting so much shared memory that you severely limit your Streaming Multiprocessor (SM) occupancy.

#### op_13: Host Allocation Data Alignment

When profiling your code in Nsight Compute (NCU), you might encounter a frustrating scenario: the tool reports "L2 sector global excess" (meaning your global memory reads are uncoalesced), even though you are absolutely certain your thread block indices and matrix dimensions are perfectly aligned. 

If your indexing math is perfect but the hardware still registers uncoalesced memory traffic, the problem often lies in *where the memory allocation actually starts*. The GPU fetches global memory in hardware-aligned chunks (typically 32-byte L2 cache lines). If the starting pointer of your data array lands in the middle of a cache line, a perfectly coalesced warp read will straddle two physical cache lines, forcing the GPU to fetch double the memory it actually needs. 

Take a close look at how the device pointers for the activation tensors are allocated in the host code. The codebase uses a single large memory allocation and mathematically partitions it into smaller chunks for the different tensors. Think about what happens to the starting memory address of tensor *N+1* if the total byte size of tensor *N* isn't perfectly aligned to a hardware-friendly boundary. 

To earn credit for this optimization, you must modify the host-side allocation logic to enforce strict memory alignment boundaries (hint: experiment with different alignment sizes) and prove the elimination of the NCU memory warnings in your final report.

#### op_14: Basic CUTLASS Device GEMM

*(Note: You may only select one CUTLASS optimization to implement for credit)*.

CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix multiplications. For this optimization, you will replace your standard GEMM or cuBLAS calls with CUTLASS's pre-defined device-level GEMM templates (e.g., `cutlass::gemm::device::Gemm`). 

To earn credit, you cannot simply copy-paste a default template. You must properly configure the `ThreadblockShape`, `WarpShape`, and `InstructionShape` parameters to specifically target the architecture of our A40 GPUs, and justify your parameter choices in your final report based on your profiling data.

Check out the [CUTLASS Documentation Hub](https://docs.nvidia.com/cutlass/4.4.2/overview.html) and flip through the version of CUTLASS and different optimizations that you want to explore here!

#### op_15: Custom Kernels with CUTLASS CuTe

*(Note: You may only select one CUTLASS optimization to implement for credit)*.

CUTLASS 3.x introduced **CuTe**, a powerful C++ template library for defining and manipulating hierarchical layouts of threads and data. While basic CUTLASS relies on pre-built `cutlass::gemm::device` templates, this optimization requires you to drop down a level of abstraction. 

For this optimization, you will build a custom kernel utilizing CuTe's `Tensor` and `Layout` objects. You must explicitly define how your memory tiles map to the threads, and orchestrate the MMA (Matrix Multiply-Accumulate) atoms manually. 

A good resource to begin learning CuTe may be the [CuTe Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html).

To earn credit, your final report must include a detailed breakdown of your custom CuTe layouts, an explanation of how your thread-to-data mapping works, and why this layout is efficient for the A40 architecture. 

#### op_16: CUTLASS Fused Epilogue (Matmul + Bias + GeLU)

*(Note: You may only select one CUTLASS optimization to implement for credit)*.

Global memory bandwidth is a massive bottleneck in large language models. While basic kernels can often handle fusing a Bias addition to a Matmul, applying a non-linear activation function typically forces the GPU to write the intermediate results to global memory and launch a completely separate kernel just to apply the activation.

For this optimization, you will configure a CUTLASS GEMM pipeline to include a **Fused Epilogue that incorporates the GeLU activation**. This allows the Streaming Multiprocessor to compute the Matmul, add the bias vector, and apply the complex GeLU mathematics directly in registers or shared memory *before* ever writing the final result back to global memory. 

You may either choose to use [GeLU Linear Combination](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/thread/linear_combination_gelu.h) from Cutlass 2.x or [Epilogue Visitor Trees](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) from Cutlass 3.x.

To earn credit, your final report must include profiling data that explicitly demonstrates the elimination of the standalone GeLU kernel launch and the corresponding reduction in global memory traffic (DRAM read/writes) compared to your un-fused baseline implementation.