# ascendc-2024-final
昇腾AI创新大赛-算子挑战赛2024年度冠军赛

## 生成工程

```bash
cd gcd
msopgen gen -i gcd.json -c ai_core-ascend910b -lan cpp -out gcd/
```

> 修改 `CMakePresets.json` 下 `ASCEND_COMPUTE_UNIT` 为 `ascend910b`

> 修改 `op_host/xxx.cpp` 下的 `this->AICore().AddConfig("ascend910");` 为 `this->AICore().AddConfig("ascend910b");`