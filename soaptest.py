import dace

from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis
from dace.transformation.estimator.soap.utils import get_lead_term
import sympy
import torch.nn.functional as F

# Launch matlab using: /home/akanksha/bin/matlab -nosplash -r "cd '/home/akanksha/Downloads/matlab'; BackgroundSolver();exit"
def soap_analysis(sdfg: dace.SDFG):
    result = perform_soap_analysis(sdfg, generate_schedule=False, solver_timeout=60)
    print("Result printing starts")
    print(result)
    print("Result printing ends")
    Q = get_lead_term(result.Q)

    # "Ss": max elements in fast memory
    # Example values! Iterate over arrays in SDFG to determine data type
    bytes_per_element = 4.0
    cache_size = 1024 * 1024
    num_elements = int(cache_size / bytes_per_element)
    
    # SOAP messes with the symbols in the SDFG, e.g., changes the case
    symbol_map = {"Ss": num_elements}
    symbol_map['d_outheight'] = outheight
    symbol_map['d_outwidth'] = outwidth
    symbol_map['d_outdepth'] = outdepth
    symbol_map['d_inchannels'] = inchannels
    symbol_map['d_outchannels'] = outchannels
    symbol_map['d_batchsize'] = batchsize
    symbol_map['d_kdim'] = kdim
    for sym in Q.free_symbols:
        print(sym)
        if str(sym) in sdfg.constants:
            symbol_map[sym] = sdfg.constants[str(sym)]
            continue

        s = str(sym).upper()
        if s in sdfg.constants:
            symbol_map[sym] = sdfg.constants[s]
    
    print(f"AB: symbol map is: {symbol_map}")
    # Now: symbol map contains all known symbol values
    # Try to get the actual value
    print(f"AB: Q is {Q}")
    simplified_Q = sympy.simplify(Q, symbols=symbol_map)
    Q_ = dace.symbolic.evaluate(simplified_Q, symbols=symbol_map)
    return Q_