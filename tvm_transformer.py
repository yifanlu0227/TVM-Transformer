import torch
import torch.nn as nn
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

###############################
# change your config here
# you should specify the cpu model using "-mcpu=" argument
# or just delete "-mcpu=znver3"
src_shape = (10,32,512)
tgt_shape = (20,32,512)
n_trails = 2000   		# higher is better. 
n_early_stopping = 600		# higher is better. 
target = "llvm -mcpu=znver3"
##############################

transformer_model = nn.Transformer(nhead=16,num_encoder_layers=12)
transformer_model.eval()

np.random.seed(2333)
src = np.random.uniform(size=src_shape).astype("float32") # shape (S,N,E)
np.random.seed(2444)
tgt = np.random.uniform(size=tgt_shape).astype("float32") # shape (T,N,E)


src_tensor = torch.tensor(src)
tgt_tensor = torch.tensor(tgt)

pytorch_output = transformer_model(src_tensor,tgt_tensor) # shape (T,N,E)
# print(pytorch_output)

dummy_input = [src_tensor,tgt_tensor]
traced_model = torch.jit.trace(transformer_model, dummy_input)
traced_model.eval()


script_module = traced_model
input_infos = [("src",(src_shape,"float32")),("tgt",(tgt_shape,"float32"))]
mod, params = relay.frontend.from_pytorch(script_module, input_infos)

#######################################
# compile on cpu
print("############################")
print("Deploy on CPU, build the relay.")

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

######################################
# TVM runtime 
print("#############################")
print("TVM runtime")
dtype = "float32"


module.set_input("src",src,tgt=tgt)

module.run()
output_shape = tgt_shape
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

# print(tvm_output)


######################################
# Collect Basic Performance Data
# ------------------------------
# We want to collect some basic performance data associated with this
# unoptimized model and compare it to a tuned model later. To help account for
# CPU noise, we run the computation in multiple batches in multiple
# repetitions, then gather some basis statistics on the mean, median, and
# standard deviation.
import timeit
print("############################")
print("measure unoptimized performance in ms")

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)


######################################
# Tune the model
# In the simplest form, tuning requires three things:
#
# - the target specification of the device you intend to run this model on
# - the path to an output file in which the tuning records will be stored
# - a path to the model to be tuned.
#

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import RandomTuner
from tvm import autotvm

print("##############################")
print("Auto Tuning CPU")

number = 4
repeat = 3
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

tuning_option = {
    "tuner": "random",
    "trials": n_trails,
    "early_stopping": n_early_stopping,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "transformer-cpu-autotuning.json",
}

tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    # XGBTuner will generate bug in this case, use RandomTuner
    # Refer to https://discuss.tvm.apache.org/t/autotvm-bug-bitserial-dense-arm-cpu/8429
    tuner_obj = RandomTuner(task) 
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )
    
    
#######################################
# Compiling an Optimized Model with Tuning Data
# ----------------------------------------------
#
# As an output of the tuning process above, we obtained the tuning records
# stored in json. The compiler will use the results to
# generate high performance code for the model on your specified target.
#
# Now that tuning data for the model has been collected, we can re-compile the
# model using optimized operators to speed up our computations.

with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

module.set_input("src",src,tgt=tgt)

module.run()
output_shape = tgt_shape
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

# print(tvm_output)

#####################################
# Comparing the Tuned and Untuned Models
print("###############################")
print("measure optimized performance in ms.")

timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}

print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))

with open("result.txt","a") as f:
    f.write("transformer cpu:\n")
    f.write("optimized: %s\n" % (optimized))
    f.write("unoptimized: %s\n" % (unoptimized))