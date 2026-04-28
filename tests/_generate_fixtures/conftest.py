# Tell pytest to skip the whole `_generate_fixtures` directory.
# These scripts are only run manually to refresh reference data.
collect_ignore_glob = ["*"]
