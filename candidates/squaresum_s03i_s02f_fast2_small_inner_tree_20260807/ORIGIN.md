# S03I origin

- Baseline: S03H, itself derived from S02F.
- Change: add the power-of-two middle-reduction tree used in S02AY while keeping
  S02F `fastPath3` unchanged.
- Cloud result: FP16/BF16 target points improved about 24%–41%; FP32 only
  120/128 correct because the long-key finalizer buffer remained 32 bytes.
- Decision: rejected, never packaged or submitted.
