import dace
import numpy as np
from dace import dtypes

from dace.transformation.interstate import StateFusion

# Define symbolic sizes for arbitrary inputs
d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')
d_kdim = dace.symbol('d_kdim')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32


# Optimize code on the GPU
def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    sdfg.apply_transformations_repeated(StateFusion)
    sdfg.simplify()
    sdfg.apply_gpu_transformations()
    return
    # Expand the maps
    m_expandparams = find_map_by_param(sdfg, 'd')
    MapExpansion.apply_to(sdfg, map_entry=m_expandparams)
    # # Collapse the maps grouped appropriately
    m_d = find_map_by_param(sdfg, 'd')
    m_h = find_map_by_param(sdfg, 'h')
    MapCollapse.apply_to(sdfg, outer_map_entry=m_d, inner_map_entry=m_h)
    m_d = find_map_by_param(sdfg, 'd')
    m_w = find_map_by_param(sdfg, 'w')
    MapCollapse.apply_to(sdfg, outer_map_entry=m_d, inner_map_entry=m_w)
    m_d = find_map_by_param(sdfg, 'd')
    m_d.map.schedule=dace.ScheduleType.GPU_Device
    m_n = find_map_by_param(sdfg, 'n')
    m_n.map.schedule=dace.ScheduleType.Sequential
    MapInterchange.apply_to(sdfg, outer_map_entry=m_n, inner_map_entry=m_d)
    m_n = find_map_by_param(sdfg, 'n')
    m_oc = find_map_by_param(sdfg, 'oc')
    MapCollapse.apply_to(sdfg, outer_map_entry=m_n, inner_map_entry=m_oc)    
    
    return
    
    # Apply tiling for the topmost map
    entry = find_map_by_param(sdfg, 'd')
    divides_evenly = True # TODO: Parameterize this
    xfutil.tile(sdfg, entry, divides_evenly, True, d=4, h=4, w=4)
    gtile_d = find_map_by_param(sdfg, 'tile_d')
    gtile_h = find_map_by_param(sdfg, 'tile_h')
    gtile_d.map.schedule = dace.ScheduleType.Sequential
    MapCollapse.apply_to(sdfg, outer_map_entry=gtile_d, inner_map_entry=gtile_h)
    gtile_d = find_map_by_param(sdfg, 'tile_d')
    gtile_w = find_map_by_param(sdfg, 'tile_w')
    MapCollapse.apply_to(sdfg, outer_map_entry=gtile_d, inner_map_entry=gtile_w)
    gtile_d = find_map_by_param(sdfg, 'tile_d')
    gtile_d.map.schedule = dace.ScheduleType.GPU_Device
    m_n = find_map_by_param(sdfg, 'n')
    m_n.map.schedule = dace.ScheduleType.GPU_ThreadBlock
    
    # mapname = 'conv3D_dace_conv3d_130_d'
    # for xform in Optimizer(sdfg).get_pattern_matches(patterns=[MapTiling]):
    #    print('Match:', xform.print_match(sdfg))
    #    matches = xform.print_match(sdfg)
    #    nameconv = re.match(r'MapTiling in \[MapEntry \((.*)_d\[d=0:d_indepth - 2, h=0:d_inheight - 2, w=0:d_inwidth - 2\].*', matches, flags=0)
    #    if(nameconv):
    #     mapname = f'{nameconv.group(1)}_d'
    #     break
    
    
    # state = sdfg.node(0)
    # conv_exit = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapExit) and n.label == mapname)
    # conv_entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.label == mapname)
    # MapTiling.apply_to(sdfg, map_entry = conv_entry, map_exit = conv_exit)
    # m_d = find_map_by_param(sdfg, 'd')
    # m_d.map.schedule = dace.ScheduleType.GPU_ThreadBlock
    # m_tiled = find_map_by_param(sdfg, 'tile_d')
    # m_tiled.map.schedule = dace.ScheduleType.GPU_Device
    # m_n = find_map_by_param(sdfg, 'n')
    # m_n.map.schedule = dace.ScheduleType.GPU_ThreadBlock
    return

# Simple parallel 3D convolution. Direct convolution
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d( Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    for n, d, h, w, oc in dace.map[0:d_batchsize, 0:d_outdepth, 0:d_outheight, 0:d_outwidth, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        for kd, kh, kw, ic in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, d+kd, h+kh, w+kw, ic] * kernel[oc, kd, kh, kw, ic]
        Output[n, d, h, w, oc] = r_tmp
