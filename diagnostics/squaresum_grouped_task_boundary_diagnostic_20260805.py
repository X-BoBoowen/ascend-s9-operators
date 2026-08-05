import json

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
AXES = (0, 2, 4)


def main():
    for output_width in (64, 320, 328):
        input_cpu = torch.full(
            (8, 1, 16, output_width, 16),
            0.5,
            dtype=torch.float16,
        )
        expected = torch.sum(
            torch.square(input_cpu), dim=AXES
        )
        actual = square_sum_v1_validation_lib.square_sum_v1(
            input_cpu.npu(),
            AXES,
            False,
            list(expected.shape),
        ).cpu()
        difference = actual.float() - expected.float()
        wrong = torch.nonzero(
            difference.abs() > 0,
            as_tuple=False,
        ).flatten()
        print(
            json.dumps(
                {
                    "output_width": output_width,
                    "outputs": actual.numel(),
                    "wrong": wrong.numel(),
                    "wrong_indices_first32": wrong[:32].tolist(),
                    "actual_first24": [
                        float(item)
                        for item in actual.flatten()[:24]
                    ],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
