#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace opengigaflow {

void puff_advantage_row(float* values, float* rewards, float* dones,
        float* advantages, float gamma, float lambda, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float delta = rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t];
        lastpufferlam = delta + gamma*lambda*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void puff_advantage_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        TORCH_CHECK(t.is_contiguous(), "All tensors must be contiguous");
    }
}


// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* dones,
        float* advantages, float gamma, float lambda,
        int num_steps, const int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        puff_advantage_row(values + offset, rewards + offset,
            dones + offset, advantages + offset,
            gamma, lambda, horizon
        );
    }
}


void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor advantages,
        double gamma, double lambda) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    puff_advantage_check(values, rewards, dones, advantages, num_steps, horizon);
    puff_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, lambda, num_steps, horizon
    );
}

TORCH_LIBRARY(opengigaflow, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) advantages, float gamma, float lambda) -> ()");
 }

TORCH_LIBRARY_IMPL(opengigaflow, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

}
