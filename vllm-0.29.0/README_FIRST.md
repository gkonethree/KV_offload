# Important before applying anything

The uploaded `vllm_offload.patch` is not a real unified git diff: it is a
pseudo-patch containing insertion instructions and `...existing code...`.
Therefore it cannot be safely transformed into a verified `git apply` patch
without the actual vLLM checkout at the target commit.

This ZIP contains the corrected package plus the exact V2 integration code and
hook contract. The original pseudo-patch is retained as
`original_vllm_offload.patch` only for comparison.

Target commit confirmed from the supplied context: `9a35c081e80a94828af6f611525102bb70e3c67f`.
