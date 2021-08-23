from transformers import BertModel, BertTokenizer, BertConfig
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda")

# Tokenizing input text
sentence_a = "Who was Jim Henson ?"
sentence_b = "Jim Henson was a puppeteer."
tokenized_text = tokenizer(sentence_a,sentence_b,padding='max_length')

# Masking one of the input tokens
input_ids = tokenized_text['input_ids']
atten_mask = tokenized_text['attention_mask']
token_type_ids = tokenized_text['token_type_ids']


# Creating a dummy input
input_ids_tensor = torch.tensor([input_ids])
atten_mask_tensors = torch.tensor([atten_mask])
token_type_ids_tensors = torch.tensor([token_type_ids])

dummy_input = [input_ids_tensor, atten_mask_tensors,token_type_ids_tensors] 


# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(torchscript=True)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, dummy_input)
traced_model.eval()
# torch.jit.save(traced_model, "traced_bert.pt")

# TVM part
# Import the graph to Relay
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor


script_module = traced_model
input_infos = [("input_ids",((1,512),"int")),("attention_mask",((1,512),"int")),("token_type_ids",((1,512),"int"))]
mod, params = relay.frontend.from_pytorch(script_module, input_infos)

#######################################
# compile on cuda
print("############################")
print("Deploy on CUDA, build the relay.")
target = tvm.target.cuda()

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

######################################
# TVM runtime 
print("#############################")
print("TVM runtime")
dtype = "float32"

input_ids = np.array(input_ids)
input_ids = input_ids[np.newaxis,...]
atten_mask = np.array(atten_mask)
atten_mask = atten_mask[np.newaxis,...]
token_type_ids = np.array(token_type_ids)
token_type_ids = token_type_ids[np.newaxis,...]

module.set_input("input_ids", input_ids,attention_mask=atten_mask,token_type_ids=token_type_ids)

module.run()
output_shape1 = (1, 512, 768)
output_shape2 = (1, 768)
tvm_output1 = module.get_output(0, tvm.nd.empty(output_shape1)).numpy()
tvm_output2 = module.get_output(1, tvm.nd.empty(output_shape2)).numpy()


######################################
# Collect Basic Performance Data
# ------------------------------
# We want to collect some basic performance data associated with this
# unoptimized model and compare it to a tuned model later. 
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
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

print("##############################")
print("Auto Tuning cuda")

target = tvm.target.cuda()

number = 20
repeat = 3
min_repeat_ms = 150  	# Using min_repeat_ms can dynamically adjusts number, so it is recommended. 
			#The typical value for NVIDIA GPU is 150 ms.
timeout = 4  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
)

tuning_option = {
    "tuner": "xgb",
    "trials": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "bert-cuda-autotuning.json",
}

tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
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


