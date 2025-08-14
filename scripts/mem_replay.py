import pynvml
import paddle
import torch
import time
pynvml.nvmlInit()
import re
import os



def set_env(is_comm = False):
    if is_comm:
        paddle.set_flags({'FLAGS_large_pool_auto_growth_chunk_size_in_mb': 500})
        paddle.set_flags({'FLAGS_small_pool_auto_growth_chunk_size_in_mb': 20})
        paddle.set_flags({'FLAGS_small_pool_size_in_mb': 10})
    else:
        paddle.set_flags({'FLAGS_large_pool_auto_growth_chunk_size_in_mb': 500})
        paddle.set_flags({'FLAGS_small_pool_auto_growth_chunk_size_in_mb': 20})
        paddle.set_flags({'FLAGS_small_pool_size_in_mb': 1})
        paddle.set_flags({'FLAGS_large_pool_pre_alloc_in_mb': 61440})
        paddle.set_flags({'FLAGS_small_pool_pre_alloc_in_mb': 500})

def unset_env():
    paddle.set_flags({'FLAGS_large_pool_auto_growth_chunk_size_in_mb': 0})
    paddle.set_flags({'FLAGS_small_pool_auto_growth_chunk_size_in_mb': 0})
    paddle.set_flags({'FLAGS_small_pool_size_in_mb': 1})
    # paddle.set_flags({'FLAGS_large_pool_pre_alloc_in_mb': 0})
    # paddle.set_flags({'FLAGS_small_pool_pre_alloc_in_mb': 0})


def process_line(line):
    pattern = r'Allocator\sinstance:\s(0x[0-9a-fA-F]+)\s(Alloc|Free)\s(\d+)\sbytes,\sptr\s=\s(0x[0-9a-fA-F]+)'
    match = re.search(pattern, line)
    if match:
        plc = match.group(1)
        cmd = match.group(2)
        size = match.group(3)
        ptr = match.group(4)
        return f"{plc} {cmd} {size} {ptr}"
    return None

def process_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            processed_line = process_line(line)
            if processed_line:
                results.append(processed_line)
    return results

def process_string(list):
    results = []
    for line in list:
        line = line.strip()
        if not line:
            continue  
        parts = line.split(' ')
        if len(parts) == 4:
            plc = parts[0].strip()
            op = parts[1].strip()
            size = parts[2].strip()
            ptr = parts[3].strip()
            try:
                size = int(size)
            except ValueError:
                pass
            results.append((plc, op, size, ptr))
    return results

def precess_allocator_stream(cmds):
    plc_bank = {}
    streams = {}
    for plc, op, size, ptr in cmds:
        plc_bank[plc] = plc_bank.get(plc, 0) + 1
    
    sorted_plc_bank = dict(sorted(plc_bank.items(), key=lambda item: item[1], reverse=True))

    for idx, plc in enumerate(sorted_plc_bank):
        if idx == 0:
            set_env(is_comm = False)
            print("plc:",plc,flush=True)
            streams[plc] = paddle.device.cuda.current_stream()
            unset_env()
        elif idx == 1:
            streams[plc] = paddle.device.cuda.Stream()
            with paddle.device.cuda.stream_guard(streams[plc]):
                set_env(is_comm = True)
                x = paddle.empty([int(1 * 1024 * 1024 * 1024)], dtype=paddle.uint8)
                del x
                unset_env()
        else:
            streams[plc] = paddle.device.cuda.Stream()

    return streams

def test_paddle(cmds,streams=None):
    set_env(is_comm = False)
    params = {}
    streams = {} if streams is None else streams
    for place, cmd, size, name in cmds:
        if place not in streams:
            print("unreachable branch")
            streams[place] = paddle.device.cuda.Stream()

        paddle.device.synchronize()
        paddle_reserved1 = paddle.device.cuda.memory_reserved() // (1024**2)
        paddle_allocated1 = paddle.device.cuda.memory_allocated() // (1024**2)
        
        with paddle.device.cuda.stream_guard(streams[place]):
            if cmd == "Alloc":
                params[name] = paddle.randn([int(int(size)/4)], dtype='float32')
            if cmd == "Free":
                del params[name]
    
        paddle.device.synchronize()
        paddle_reserved2 = paddle.device.cuda.memory_reserved() // (1024**2)
        paddle_allocated2 = paddle.device.cuda.memory_allocated() // (1024**2)

        print("reserved1 = ", paddle_reserved1, "reserved2 = ", paddle_reserved2,
            "allocated1 = ", paddle_allocated1, "allocated2 = ", paddle_allocated2,
            "auto growth = ", paddle_reserved2 - paddle_reserved1)
    print(len(streams))


def test_torch(cmds):
    params = {}
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    for cmd, size, name in cmds:

        torch_reserved1 = torch.cuda.memory_reserved() // (1024** 2)
        torch_allocated1 = torch.cuda.memory_allocated() // (1024** 2)

        if cmd == "Alloc":
            params[name] = torch.rand(int(int(size)/4))
        if cmd == "Free":
            del params[name]

        torch_reserved2 = torch.cuda.memory_reserved() // (1024** 2)
        torch_allocated2 = torch.cuda.memory_allocated() // (1024** 2)

        print("reserved1 = ", torch_reserved1, "reserved2 = ", torch_reserved2,
            "allocated1 = ", torch_allocated1, "allocated2 = ", torch_allocated2,
            "auto growth = ", torch_reserved2 - torch_reserved1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python test_ds.py <input_file> <torch/paddle>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    is_paddle = sys.argv[2]

    output = process_file(input_file)
    output = process_string(output)

    print("stream:",precess_allocator_stream(output),flush=True)

    if is_paddle == "paddle":
        test_paddle(output, precess_allocator_stream(output))
    else:
        # torch.cuda.memory._record_memory_history()
        test_torch(output)
        # torch.cuda.memory._dump_snapshot("torch_ds_8nodes.pickle")