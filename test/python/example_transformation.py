from approxMLIR import approxMLIR
args = None
def untransformed_kernel():
    pass

def qauntize_to_quint8():
    pass

def quantize_to_fp16():
    pass

def quantize_to_fp32():
    pass
states = []
threshold = []
num_thresholds = []

untransformed_kernel()

def strategy_1():
    if states[0] <= threshold[0][0]:
        qauntize_to_quint8()
    elif states[0] <= threshold[0][1]:
        quantize_to_fp16()
    elif states[0] <= threshold[0][2]:
        quantize_to_fp32()
    else:
        untransformed_kernel()
        

    knobs = [
        # error knob 1
        num_thresholds[0], # also tunable
        [threshold[0][0], qauntize_to_quint8,
        threshold[0][1], quantize_to_fp16, 
        threshold[0][2], quantize_to_fp32],
        # error knob 2 
        num_thresholds[1],
        [threshold[1][0], quantize_to_fp16, 
            threshold[1][1], quantize_to_fp32,
        threshold[1][2], quantize_to_fp16]
        # for each knob, > threshold[knob][-1] 
        # is set to untransformed_kernel
    ]
    
def strategy_2(approxMLIR, states):
    # for error knob 0, check if we want to turn on error knob
    if approxMLIR.switch(0, states[0]):
        quantize_to_fp16()
    else: # recover from error
        untransformed_kernel()



