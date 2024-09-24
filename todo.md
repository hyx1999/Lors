# TODO

TODO:
1. Main:

- Llama-2-7b wanda-2:4 (test)
- Llama-2-7b sparsegpt-2:4 (test)
- Llama-2-13b wanda-2:4 (test)
- Llama-2-13b sparsegpt-2:4 (test)
- Llama-3-8b wanda-2:4
- Llama-3-8b sparsegpt-2:4

- pretrain data:
    - general domain:
        - Llama-2-7b wanda-2:4 lora_rank16 0.5B lr1e-3 
            - sp-lora (train, test)
            - lora (train, test)
            - spp
        - Llama-2-7b sparsegpt-2:4 lora_rank16 0.5B lr1e-3
            - sp-lora (train, test)
            - lora (train, test)
            - spp
        - Llama-2-13b wanda-2:4 lora_rank16 0.5B lr1e-3
            - sp-lora (train, test)
            - lora (train, test)
            - spp
        - Llama-2-13b sparsegpt-2:4 lora_rank16 0.5B lr1e-3
            - sp-lora (train, test)
            - lora (train, test)
            - spp

        - Llama-3-8b wanda-2:4 lora_rank16 0.5B lr1e-3
            - sp-lora (train)
            - lora (train)
            - spp

        - Llama-3-8b sparsegpt-2:4 lora_rank16 0.5B lr1e-3
            - sp-lora (train)
            - lora (train)
            - spp

- sft data:
    - general domain:
        - Llama-2-7b wanda-2:4 lora_rank16 alpaca lr1e-3
            - sp-lora (train, test)
            - lora (train)
            - spp (train, test)
        - Llama-2-7b sparsegpt-2:4 lora_rank16 alpaca lr1e-3
            - sp-lora (train, test)
            - lora (train)
            - spp

        - Llama-2-13b wanda-2:4 lora_rank16 alpaca lr1e-3
            - sp-lora (train, test)
            - lora (train)
            - spp

        - Llama-2-13b sparsegpt-2:4 lora_rank16 alpaca lr1e-3
            - sp-lora (train, test)
            - lora (train)
            - spp

        - Llama-3-8b wanda-2:4 lora_rank16 alpaca lr1e-3
            - sp-lora (train, test)
            - lora (train)
            - spp

        - Llama-3-8b sparsegpt-2:4 lora_rank16 alpaca lr1e-3
            - sp-lora (train, test)
            - lora (train)
            - spp

    - specific domain: 
         <!-- train dataset -->
        <!-- chat: 52k subsetof WizardLM -->
        <!-- math: 100k subset of MetaMathQA -->
        <!-- Code: 100k subset of Code-Feedback -->
        <!-- test dataset -->
        <!-- MT-Bench, GSM8K, Human-eval (follow lora-ga) -->

        - Llama-2-7b wanda-2:4
            - sp-lora
            - lora
            - spp

        - Llama-3-8b wanda-2:4 
            - sp-lora
            - lora
            - spp

2. Ablation Study:

- Llama-2-7b wanda-2:4 lora_rank16 0.5B (train, test)
- Llama-2-7b wanda-2:4 lora_rank16 1B (train, test)
- Llama-2-7b wanda-2:4 lora_rank16 2B
- Llama-2-7b wanda-2:4 lora_rank128 0.5B (train, test)
- Llama-2-7b wanda-2:4 lora_rank128 1B (train, test)
- Llama-2-7b wanda-2:4 lora_rank128 2B
