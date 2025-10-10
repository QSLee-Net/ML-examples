/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its
 * affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/extension/llm/tokenizers/include/pytorch/tokenizers/sentencepiece.h>

#include <chrono>
#include <random>
#include <fstream>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::runtime::TensorInfo;
using executorch::runtime::etensor::Tensor;
using executorch::extension::randn;
using executorch::extension::randint;

constexpr size_t k_seed_default = 99;
constexpr size_t k_num_steps_default = 8;
constexpr size_t k_audio_len_sec_default = 10.0f;

// -- Update the tensor index based on your model configuration.
constexpr size_t k_t5_ids_in_idx = 0;
constexpr size_t k_t5_attnmask_in_idx = 1;
constexpr size_t k_t5_audio_len_in_idx = 2;

constexpr size_t k_dit_t_in_idx = 1;

// -- Fill sigmas params
constexpr float k_logsnr_max = -6.0f;
constexpr float k_sigma_min = 0.0f;
constexpr float k_sigma_max = 1.0f;

#define AUDIOGEN_CHECK(x)                                 \
    if (!(x)) {                                                 \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);\
        exit(1);                                                \
    }

static inline long time_in_ms() {
    using namespace std::chrono;
    auto now = time_point_cast<milliseconds>(steady_clock::now());
    return now.time_since_epoch().count();
}

static void print_usage(const char *name) {
    fprintf(stderr,
        "Usage: %s -m <models_base_path> -p <prompt> -t <num_threads> [-s <seed> -l <audio_len>]\n\n"
        "Options:\n"
        "  -m <models_base_path>   Path to model files\n"
        "  -p <prompt>             Input prompt text (e.g., warm arpeggios on house beats 120BPM with drums effect)\n"
        "  -t <num_threads>        Number of CPU threads to use\n"
        "  -s <seed>               (Optional) Random seed for reproducibility. Different seeds generate different audio samples (Default: %zu)\n"
        "  -l <audio_len_sec>      (Optional) Length of generated audio (Default: %zu s)\n"
        "  -n <num_steps>          (Optional) Number of steps (Default: %zu)\n"
        "  -o <output_file>        (Optional) Output audio file name (Default: <prompt>_<seed>.wav)\n"
        "  -d <dummy_run>          (Optional) Run a dummy run to warm up the model (Default: false)\n"
        "  -h                      Show this help message\n",
        name,
        k_seed_default,
        k_audio_len_sec_default,
        k_num_steps_default);
}

static std::string get_filename(std::string prompt, size_t seed) {
    // Convert spaces to underscores
    std::replace(prompt.begin(), prompt.end(), ' ', '_');

    // Convert to lowercase
    std::transform(prompt.begin(), prompt.end(), prompt.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return prompt + "_" + std::to_string(seed) + ".wav";
}

static void fill_random_norm_dist(float* buff, size_t buff_sz, size_t seed) {
    std::random_device rd{};
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);

    auto gen_fn = [&dis, &gen](){ return dis(gen); };
    std::generate(buff, buff + buff_sz, gen_fn);
}

static void fill_sigmas(std::vector<float>& arr, float start, float end) {

    const int32_t sz = static_cast<int32_t>(arr.size());
    const float step = ((end - start) / static_cast<float> (sz - 1));

    // Linspace
    arr[0]      = start;
    arr[sz - 1] = end;

    for(int32_t i = 1; i < sz - 1; ++i) {
        arr[i] = arr[i - 1] + step;
    }

    for(int32_t i = 0; i < sz; ++i) {
        arr[i] = 1.0f / (1.0f + std::exp(arr[i])) ;
    }

    arr[0]      = k_sigma_max;
    arr[sz - 1] = k_sigma_min;
}

static void sampler_ping_pong(float* dit_out_data, float* dit_x_tensor, size_t dit_x_in_sz, float cur_t, float next_t, size_t step_idx, size_t seed) {

    for(size_t i = 0; i < dit_x_in_sz; i++) {
        dit_out_data[i] = dit_x_tensor[i] - ( cur_t * dit_out_data[i]);
    }

    std::vector<float> rand_tensor(dit_x_in_sz);
    fill_random_norm_dist(rand_tensor.data(), dit_x_in_sz, seed);

    // x = (1-t_next) * denoised + t_next * torch.randn_like(x)
    for(size_t i = 0; i < dit_x_in_sz; i++) {
        dit_x_tensor[i] = ((1.0f - next_t) * dit_out_data[i]) + (next_t * rand_tensor[i]);
    }
}

static void save_as_wav(const std::string& path, const float* left_ch, const float* right_ch, size_t buffer_sz) {
    constexpr int32_t audio_sr = 44100;
    constexpr int32_t audio_num_channels = 2;
    constexpr int32_t audio_bits_per_sample = 32;
    constexpr uint16_t audio_format = 3; // IEEE float

    const int32_t byte_rate = audio_sr * audio_num_channels * (audio_bits_per_sample / 8);
    const int32_t block_align = audio_num_channels * (audio_bits_per_sample / 8);
    const int32_t data_chunk_sz = buffer_sz * 2 * sizeof(float);
    const int32_t fmt_chunk_sz = 16;
    const int32_t header_sz = 44;
    const int32_t file_sz = header_sz + data_chunk_sz - 8;

    std::ofstream out_file(path, std::ios::binary);

    // Prepare the header
    // RIFF header
    out_file.write("RIFF", 4);
    out_file.write(reinterpret_cast<const char*>(&file_sz), 4);
    out_file.write("WAVE", 4);
    out_file.write("fmt ", 4);
    out_file.write(reinterpret_cast<const char*>(&fmt_chunk_sz), 4);
    out_file.write(reinterpret_cast<const char*>(&audio_format), 2);
    out_file.write(reinterpret_cast<const char*>(&audio_num_channels), 2);
    out_file.write(reinterpret_cast<const char*>(&audio_sr), 4);
    out_file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out_file.write(reinterpret_cast<const char*>(&block_align), 2);
    out_file.write(reinterpret_cast<const char*>(&audio_bits_per_sample), 2);

    // Store the data in interleaved format (L0, R0, L1, R1,....)
    out_file.write("data", 4);
    out_file.write(reinterpret_cast<const char*>(&data_chunk_sz), 4);

    for (size_t i = 0; i < buffer_sz; ++i) {
        out_file.write(reinterpret_cast<const char*>(&left_ch[i]), sizeof(float));
        out_file.write(reinterpret_cast<const char*>(&right_ch[i]), sizeof(float));
    }

    out_file.close();
}

std::vector<executorch::aten::SizesType> get_tensor_dims(const TensorInfo& tensor_info) {
    std::vector<executorch::aten::SizesType> tensor_dims(tensor_info.sizes().begin(), tensor_info.sizes().end());
    return tensor_dims;
}

size_t get_tensor_numel(const std::vector<executorch::aten::SizesType>& tensor_dims) {
    size_t numel = 1;
    for (const auto& dim : tensor_dims) {
        numel *= dim;
    }
    return numel;
}

void dry_run(std::unique_ptr<executorch::extension::Module>& module) {

    // Dummy run for a module
    // -------------------------
    auto module_forward_meta_res = module->method_meta("forward");
    AUDIOGEN_CHECK(module_forward_meta_res.ok());

    auto module_forward_meta = module_forward_meta_res.get();
    size_t module_num_inputs = module_forward_meta.num_inputs();

    // To keep the created tensors inside the loop alive
    std::vector<std::shared_ptr<Tensor>> allocated_tensors;

    std::vector<executorch::runtime::EValue> module_inputs(module_num_inputs);
    for (size_t i = 0; i < module_num_inputs; ++i) {
        auto input_tensor_meta = module_forward_meta.input_tensor_meta(i).get();
        auto input_tensor_dims = get_tensor_dims(input_tensor_meta);
        auto input_tensor_scalar_type = input_tensor_meta.scalar_type();
        auto input_tensor = input_tensor_scalar_type == ScalarType::Float ? randn(input_tensor_dims, input_tensor_scalar_type) : randint(1, 100, input_tensor_dims, input_tensor_scalar_type);
        allocated_tensors.push_back(input_tensor);
        module_inputs[i] = input_tensor;
    }

    auto module_output = module->forward(module_inputs);
    if(!module_output.ok()) {
        ET_LOG(Error, "Failed to run module forward");
        exit(EXIT_FAILURE);
    }
}

int main(int32_t argc, char** argv) {

    // Required arguments
    std::string models_base_path = "";
    std::string prompt           = "";
    size_t cpu_threads           = -1;

    // Optional arguments
    std::string output_path      = "";
    size_t seed                  = k_seed_default;
    size_t num_steps             = k_num_steps_default;
    float audio_len_sec          = static_cast<float>(k_audio_len_sec_default);
    bool  run_dummy_run          = false;

    int opt;
    while ((opt = getopt(argc, argv, "m:p:t:s:n:o:l:d:h")) != -1) {
        switch (opt) {
            case 'm': models_base_path = optarg; break;
            case 'p': prompt           = optarg; break;
            case 't': cpu_threads      = std::stoull(optarg); break;
            case 'o': output_path      = optarg; break;
            case 's': seed             = std::stoull(optarg); break;
            case 'n': num_steps        = std::stoull(optarg); break;
            case 'l': audio_len_sec    = static_cast<float>(std::stoull(optarg)); break;
            case 'd': run_dummy_run    = (std::string(optarg) == "true"); break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (models_base_path.empty() || prompt.empty() || cpu_threads <= 0) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string t5_model = models_base_path + "/conditioners_model.pte";
    std::string dit_model = models_base_path + "/dit_model.pte";
    std::string autoencoder_model = models_base_path + "/autoencoder_model.pte";
    std::string sentence_model_path = models_base_path + "/spiece.model";

#if defined(ET_USE_THREADPOOL)
    uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
    ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
    if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
    }
#else
    uint32_t num_performant_cores = 4;
#endif
    ET_LOG(Info, "Using %d threads", num_performant_cores);

    // ----- Load the models
    // ----------------------------------
    std::unique_ptr<executorch::extension::Module> t5_module;
    t5_module = std::make_unique<Module>(t5_model, Module::LoadMode::File);
    ET_LOG(Info, "Model (%s) loaded", t5_model.c_str());

    std::unique_ptr<executorch::extension::Module> dit_module;
    dit_module = std::make_unique<Module>(dit_model, Module::LoadMode::File);
    ET_LOG(Info, "Model (%s) loaded", dit_model.c_str());

    std::unique_ptr<executorch::extension::Module> autoencoder_module;
    autoencoder_module = std::make_unique<Module>(autoencoder_model, Module::LoadMode::File);
    ET_LOG(Info, "Model (%s) loaded", autoencoder_model.c_str());

    // ----- Get models forward methods meta
    // ----------------------------------
    auto t5_forward_meta_res = t5_module->method_meta("forward");
    if (!t5_forward_meta_res.ok()) {
        ET_LOG(Error, "Failed to get method meta for t5 'forward'");
        return EXIT_FAILURE;
    }
    auto t5_forward_meta = t5_forward_meta_res.get();

    auto dit_forward_meta_res = dit_module->method_meta("forward");
    if (!dit_forward_meta_res.ok()) {
        ET_LOG(Error, "Failed to get method meta for dit 'forward'");
        return EXIT_FAILURE;
    }
    auto dit_forward_meta = dit_forward_meta_res.get();

    auto autoencoder_forward_meta_res = autoencoder_module->method_meta("forward");
    if (!autoencoder_forward_meta_res.ok()) {
        ET_LOG(Error, "Failed to get method meta for autoencoder 'forward'");
        return EXIT_FAILURE;
    }
    auto autoencoder_forward_meta = autoencoder_forward_meta_res.get();

    // Load tokenizer
    // ----------------------------------
    auto tokenizer = std::make_unique<tokenizers::SPTokenizer>();
    auto err = tokenizer->load(sentence_model_path);
    if (err != tokenizers::Error::Ok) {
        ET_LOG(Error, "failed to load tokenizer");
        return EXIT_FAILURE;
    }

    // Dummy run if needed
    if (run_dummy_run) {
        ET_LOG(Info, "Running dummy forward pass for all models...");
        dry_run(t5_module);
        dry_run(dit_module);
        dry_run(autoencoder_module);
        ET_LOG(Info, "Dummy Run finished.");
    }

    // Tokenize the prompt
    auto token_result = tokenizer->encode(prompt, 0, 1);
    if (token_result.error() != tokenizers::Error::Ok) {
        ET_LOG(Error, "failed to tokenize prompt");
        return EXIT_FAILURE;
    }
    auto tokens = token_result.get();

    // ----- Prepare t5 input tensors
    // ----------------------------------

    // Prepare input_ids tensor data
    const auto t5_input_ids_tensor_dims = get_tensor_dims(t5_forward_meta.input_tensor_meta(k_t5_ids_in_idx).get());
    auto t5_seq_len = t5_input_ids_tensor_dims[1];

    std::vector<uint64_t> token_ids(t5_seq_len, 0);
    for (int i = 0; i < tokens.size(); i++) {
        token_ids[i] = static_cast<uint64_t>(tokens[i]);
    }

    // Prepare input mask tensor data
    const auto t5_input_mask_tensor_dims = get_tensor_dims(t5_forward_meta.input_tensor_meta(k_t5_attnmask_in_idx).get());
    std::vector<uint64_t> attention_mask(t5_seq_len, 0);
    for (int i = 0; i < tokens.size(); i++) {
        attention_mask[i] = 1U;
    }

    // Prepare length tensor data
    auto t5_input_len_tensor_dims = get_tensor_dims(t5_forward_meta.input_tensor_meta(k_t5_audio_len_in_idx).get());
    const size_t t5_length_in_sz = get_tensor_numel(t5_input_len_tensor_dims);
    AUDIOGEN_CHECK(t5_length_in_sz == 1);

    // Prepare T5 tensors
    auto token_ids_tensor = executorch::extension::from_blob(
        token_ids.data(), t5_input_ids_tensor_dims, ScalarType::Long);

    auto attention_mask_tensor = executorch::extension::from_blob(
        attention_mask.data(), t5_input_mask_tensor_dims, ScalarType::Long);

    auto duration_tensor = executorch::extension::from_blob(
        &audio_len_sec, t5_input_len_tensor_dims, ScalarType::Float);

    std::vector<executorch::runtime::EValue> condintioners_inputs = {   token_ids_tensor,
                                                                        attention_mask_tensor,
                                                                        duration_tensor };

    // Run t5 forward
    auto t5_start = time_in_ms();
    auto condintioners_result = t5_module->forward(condintioners_inputs);
    auto t5_end = time_in_ms();
    if (condintioners_result.error() != executorch::runtime::Error::Ok) {
        ET_LOG(Error, "failed to run t5 forward function");
        return 1;
    }

    // Get t5 output tensors
    const auto cross_attn_cond_tensor = condintioners_result->at(0).toTensor();
    const auto cross_attn_mask_tensor = condintioners_result->at(1).toTensor();
    const auto global_cond_tensor     = condintioners_result->at(2).toTensor();

    // ----- Prepare DiT input tensors
    // ----------------------------------

    // Prepare the X input tensor
    const auto dit_x_tensor_dims = get_tensor_dims(dit_forward_meta.input_tensor_meta(k_t5_ids_in_idx).get());
    const size_t x_in_sz = get_tensor_numel(dit_x_tensor_dims);

    std::vector<float> x_data(x_in_sz, 0.0f);
    fill_random_norm_dist(x_data.data(), x_data.size(), seed);

    auto x_tensor = executorch::extension::from_blob(
        x_data.data(), dit_x_tensor_dims, ScalarType::Float);
    const auto x_data_ptr = x_tensor->mutable_data_ptr<float>();

    // Prepare Sigmas values
    const auto dit_t_tensor_dims = get_tensor_dims(dit_forward_meta.input_tensor_meta(k_dit_t_in_idx).get());
    const size_t t_in_sz = get_tensor_numel(dit_t_tensor_dims);
    AUDIOGEN_CHECK(t_in_sz == 1);

    std::vector<float> t_buffer(num_steps + 1);
    fill_sigmas(t_buffer, k_logsnr_max, 2.0f);

    auto dit_start = time_in_ms();
    for(size_t i = 0; i < num_steps; ++i) {

        float curr_t = t_buffer[i];
        float next_t = t_buffer[i + 1];
        auto t_tensor = executorch::extension::from_blob(
            &curr_t, dit_t_tensor_dims, ScalarType::Float);

        std::vector<executorch::runtime::EValue> dit_inputs = {
            x_tensor,
            t_tensor,
            cross_attn_cond_tensor,
            global_cond_tensor,
        };
        auto dit_result = dit_module->forward(dit_inputs);
        if (dit_result.error() != executorch::runtime::Error::Ok) {
            ET_LOG(Error, "failed to run dit forward function");
            return 1;
        }
        // Get the output tensor
        const auto dit_x_tensor_result = dit_result->at(0).toTensor();
        auto* dit_x_data_result = dit_x_tensor_result.mutable_data_ptr<float>();

        sampler_ping_pong(dit_x_data_result, x_data_ptr, x_in_sz, curr_t, next_t, i, seed + i);
    }

    auto dit_end = time_in_ms();

    // (3) Run AutoEncoder to convert the output to waveform
    std::vector<executorch::runtime::EValue> autoencoder_inputs = { x_tensor };
    auto autoencoder_start = time_in_ms();
    auto autoencoder_result = autoencoder_module->forward(autoencoder_inputs);
    auto autoencoder_end = time_in_ms();
    if (autoencoder_result.error() != executorch::runtime::Error::Ok) {
        ET_LOG(Error, "failed to run autoencoder forward function");
        return 1;
    }

    // Save the output to Wav
    // Get the output size of autoencoder module
    const auto output_waveform_tensor = autoencoder_result->at(0).toTensor();
    const auto output_waveform_data = output_waveform_tensor.mutable_data_ptr<float>();
    const size_t output_waveform_sz_per_channel = output_waveform_tensor.numel() / 2;
    const auto left_ch = output_waveform_data;
    const auto right_ch = output_waveform_data + output_waveform_sz_per_channel;
    save_as_wav("output.wav", left_ch, right_ch, output_waveform_sz_per_channel);
    ET_LOG(Info, "Output waveform saved to output.wav");

    // Print total execution time
    auto t5_exec_time = t5_end - t5_start;
    auto dit_exec_time = dit_end - dit_start;
    auto dit_avg_step_time     = (dit_exec_time / static_cast<float>(num_steps));
    auto autoencoder_exec_time = autoencoder_end - autoencoder_start;
    auto total_exec_time = t5_exec_time + dit_exec_time + autoencoder_exec_time;

    ET_LOG(Info, "T5: %ld ms", t5_exec_time);
    ET_LOG(Info, "DiT: %ld ms", dit_exec_time);
    ET_LOG(Info, "DiT Avg per step: %f ms", dit_avg_step_time);
    ET_LOG(Info, "AutoEncoder: %ld ms", autoencoder_exec_time);
    ET_LOG(Info, "Total execution time: %ld ms", total_exec_time);
}