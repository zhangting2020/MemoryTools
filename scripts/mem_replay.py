import paddle
import torch
import time
import re
import os


def set_env():
        paddle.set_flags({'FLAGS_large_pool_auto_growth_chunk_size_in_mb': 500})
        paddle.set_flags({'FLAGS_small_pool_auto_growth_chunk_size_in_mb': 20})
        paddle.set_flags({'FLAGS_small_pool_size_in_mb': 1})
        paddle.set_flags({'FLAGS_large_pool_pre_alloc_in_mb': 61440})
        paddle.set_flags({'FLAGS_small_pool_pre_alloc_in_mb': 500})


def process_line(line):
    pattern = re.compile(
      r'Allocator\s+instance:\s*'
      r'(0x[0-9a-fA-F]+)\s+'           # allocator地址
      r'(Alloc|Free)\s+'               # operation: Alloc or Free
      r'(\d+)\s+bytes,'                # size
      r'\s*ptr\s*=\s*(0x[0-9a-fA-F]+)' # ptr
      r'(?:,\s*place\s*=\s*Place\(([^)]+)\))?'  # optional: place
    )

    match = re.search(pattern, line)
    if match:
        allocator = match.group(1)
        cmd = match.group(2)
        size = match.group(3)
        ptr = match.group(4)
        place = match.group(5) if match.group(5) else None
        return f"{allocator} {cmd} {size} {ptr} {place}"
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
            continue  # 跳过空行
        parts = line.split(' ')
        if len(parts) == 5:
            allocator = parts[0].strip()
            op = parts[1].strip()
            size = parts[2].strip()
            ptr = parts[3].strip()
            place = parts[4].strip()
            try:
                size = int(size)
            except ValueError:
                pass
            results.append((allocator, op, size, ptr, place))
    return results
    
def get_stream_for_paddle(idx):
    if idx == 0:
        stream = paddle.device.cuda.current_stream()
    elif idx == 1:
        stream = paddle.device.cuda.Stream()
        with paddle.device.cuda.stream_guard(stream):
            x = paddle.empty([int(1 * 1024 * 1024 * 1024)], dtype=paddle.uint8)
            del x
    else:
        stream = paddle.device.cuda.Stream()
    return stream

def get_stream_for_torch(idx):
    if idx == 0:
        stream = torch.cuda.current_stream()
    else:
        stream = torch.cuda.Stream()
    return stream

def precess_allocator_stream(cmds, framework):
    allocator_bank = {}
    streams = {}
    for allocator, op, size, ptr, place in cmds:
        allocator_bank[allocator] = allocator_bank.get(allocator, 0) + 1
    sorted_allocator_bank = dict(sorted(allocator_bank.items(), key=lambda item: item[1], reverse=True))

    for idx, allocator in enumerate(sorted_allocator_bank):
        print("allocator {}".format(allocator), flush=True)
        streams[allocator] = get_stream_for_paddle(idx) if framework == 'paddle' else get_stream_for_torch(idx)
    return streams

def operation_on_gpu_device(op, device):
    if op == "Free":
        return True
    elif op == "Alloc":
        return bool(re.match(r"^gpu:[0-7]$", device))
    else:
        return False

def test_paddle(cmds, streams=None):
    params = {}
    streams = {} if streams is None else streams
    for allocator, op, size, ptr, place in cmds:
        if allocator not in streams:
            print("unreachable branch")
            streams[allocator] = paddle.device.cuda.Stream()
        if not operation_on_gpu_device(op, place):
            print("op in place {} will be ignore.".format(place))
            continue

        paddle.device.synchronize()
        paddle_reserved1 = paddle.device.cuda.memory_reserved() // (1024**2)
        paddle_allocated1 = paddle.device.cuda.memory_allocated() // (1024**2)
        
        with paddle.device.cuda.stream_guard(streams[allocator]):
            if op == "Alloc":
                params[ptr] = paddle.randn([int(int(size)/4)], dtype='float32')
            if op == "Free" and ptr in params:
                del params[ptr]
    
        paddle.device.synchronize()
        paddle_reserved2 = paddle.device.cuda.memory_reserved() // (1024**2)
        paddle_allocated2 = paddle.device.cuda.memory_allocated() // (1024**2)
        paddle_max_reserved = paddle.device.cuda.max_memory_reserved() // (1024**2)
        paddle_max_allocated = paddle.device.cuda.max_memory_allocated() // (1024**2)

        print("reserved = {} allocated = {} auto growth = {} max_allocated = {} max_reserved = {}".format(
            paddle_reserved2,
            paddle_allocated2,
            paddle_reserved2 - paddle_reserved1,
            paddle_max_allocated,
            paddle_max_reserved)
        )
    print(len(streams))

def test_torch(cmds, streams=None):
    params = {}
    device = torch.device("cuda:0")
    for allocator, op, size, ptr, place in cmds:
        if allocator not in streams:
            print("unreachable branch")
            streams[allocator] = paddle.device.cuda.Stream()
        if not operation_on_gpu_device(op, place):
            print("op in place {} will be ignore.".format(place))
            continue

        torch_reserved1 = torch.cuda.memory_reserved() // (1024** 2)
        torch_allocated1 = torch.cuda.memory_allocated() // (1024** 2)

        if op == "Alloc":
            params[ptr] = torch.rand(int(int(size)/4), device=device)
        elif op == "Free" and ptr in params:
            del params[ptr]

        torch_reserved2 = torch.cuda.memory_reserved() // (1024** 2)
        torch_allocated2 = torch.cuda.memory_allocated() // (1024** 2)
        torch_max_reserved = torch.cuda.max_memory_reserved() // (1024**2)
        torch_max_allocated = torch.cuda.max_memory_allocated() // (1024**2)

        print("reserved = {} allocated = {} auto growth = {} max_allocated = {} max_reserved = {}".format(
                torch_reserved2,
                torch_allocated2,
                torch_reserved2 - torch_reserved1,
                torch_max_allocated,
                torch_max_reserved)
            )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python test_ds.py <input_file> <torch/paddle>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    is_paddle = sys.argv[2]

    output = process_file(input_file)
    output = process_string(output)
    streams = precess_allocator_stream(output, sys.argv[2])

    print("stream: {}".format(streams), flush=True)

    if is_paddle == "paddle":
        set_env()
        test_paddle(output, streams)
    else:
        #torch.cuda.memory._record_memory_history()
        test_torch(output, streams)
        #torch.cuda.memory._dump_snapshot("torch.pickle")