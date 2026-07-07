{
    trainer+: {
        actor_model+: {
            pretrained_args+: {
                use_flash_attention_2: false,
            },
        },
        critic_model+: {
            pretrained_backbone_model+: {
                pretrained_args+: {
                    use_flash_attention_2: false,
                },
            },
        },
        reference_model+: {
            pretrained_args+: {
                use_flash_attention_2: false,
            },
        },
    },
}
