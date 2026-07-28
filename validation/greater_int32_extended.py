import random

import torch
import torch_npu

import greater_validation_lib


SEED = 2026072803


def check(name, self_cpu, other_cpu):
    expected = torch.gt(self_cpu, other_cpu)
    actual = greater_validation_lib.greater(
        self_cpu.npu(),
        other_cpu.npu(),
    ).cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.nonzero(actual != expected)
        first = tuple(mismatch[0].tolist()) if mismatch.numel() else ()
        details = []
        for position in mismatch[:8]:
            index = tuple(position.tolist())
            self_value = self_cpu.expand(expected.shape)[index].item()
            other_value = other_cpu.expand(expected.shape)[index].item()
            details.append(
                (
                    index,
                    self_value,
                    other_value,
                    bool(expected[index]),
                    bool(actual[index]),
                )
            )
        raise AssertionError(
            f"{name}: mismatch={mismatch.shape[0]}, first={first}, "
            f"details={details}"
        )
    print(
        f"PASS {name}: self={tuple(self_cpu.shape)}, "
        f"other={tuple(other_cpu.shape)}, output={tuple(expected.shape)}"
    )


def random_int32(shape):
    count = 1
    for extent in shape:
        count *= extent
    values = []
    for _ in range(count):
        value = random.getrandbits(32)
        if value >= 2**31:
            value -= 2**32
        values.append(value)
    return torch.tensor(values, dtype=torch.int32).reshape(shape)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    minimum = torch.iinfo(torch.int32).min
    maximum = torch.iinfo(torch.int32).max
    boundaries = torch.tensor(
        [
            minimum,
            minimum + 1,
            -(2**30),
            -65537,
            -65536,
            -1,
            0,
            1,
            65535,
            65536,
            (2**30) - 1,
            maximum - 1,
            maximum,
        ],
        dtype=torch.int32,
    )

    check(
        "boundary_cartesian",
        boundaries.reshape(-1, 1),
        boundaries.reshape(1, -1),
    )
    check(
        "random_full_bit_pattern",
        random_int32((257, 1021)),
        random_int32((257, 1021)),
    )
    equal = random_int32((200_003,))
    check("all_equal", equal, equal.clone())
    check(
        "right_scalar_min",
        random_int32((200_003,)),
        torch.tensor(minimum, dtype=torch.int32),
    )
    check(
        "left_scalar_max",
        torch.tensor(maximum, dtype=torch.int32),
        random_int32((200_003,)),
    )
    check(
        "trailing_broadcast",
        random_int32((257, 513)),
        random_int32((513,)),
    )
    check(
        "interleaved_broadcast",
        random_int32((17, 1, 19, 1)),
        random_int32((1, 23, 1, 29)),
    )
    check(
        "rank8_broadcast",
        random_int32((2, 1, 3, 1, 2, 1, 3, 5)),
        random_int32((1, 2, 1, 3, 1, 2, 1, 5)),
    )
    check(
        "empty_output",
        torch.empty((0, 3), dtype=torch.int32),
        torch.empty((1, 3), dtype=torch.int32),
    )
    torch.npu.synchronize()
    print("ALL PASS: 9 int32 extended cases")


if __name__ == "__main__":
    main()
