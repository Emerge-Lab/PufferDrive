// Drive-only build: no env-specific encoder overrides remain.

static void create_custom_encoder(const std::string& env_name, Encoder* enc) {
    (void)env_name;
    (void)enc;
}
