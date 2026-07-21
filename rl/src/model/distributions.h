#pragma once
#include <torch/torch.h>

#include "env/interfaces.h"

namespace rl {

torch::Tensor sampleActions(const torch::Tensor& logits, const ActionSpec& spec,
                            bool deterministic);

std::pair<torch::Tensor, torch::Tensor> sampleWithLogProb(const torch::Tensor& logits,
                                                          const ActionSpec& spec,
                                                          bool deterministic);

torch::Tensor logProb(const torch::Tensor& logits, const ActionSpec& spec,
                      const torch::Tensor& actions);

std::pair<torch::Tensor, torch::Tensor> logProbEntropy(const torch::Tensor& logits,
                                                       const ActionSpec& spec,
                                                       const torch::Tensor& actions);

std::vector<double> branchEntropies(const torch::Tensor& logits, const ActionSpec& spec);

}
