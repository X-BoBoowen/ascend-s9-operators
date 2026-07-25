template <typename U>
__aicore__ inline void PrintVec(LocalTensor<U>& x, int offset, int num) {
    for (int i = 0; i < num; i++) {
        printf("%f ", x.GetValue(offset + i));
    }
    printf("\n");
}

template <typename U>
__aicore__ inline void PrintVecD(LocalTensor<U>& x, int offset, int num) {
    for (int i = 0; i < num; i++) {
        printf("%d ", x.GetValue(offset + i));
    }
    printf("\n");
}