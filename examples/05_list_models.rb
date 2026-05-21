# Walk the system for GGUF models. Scans your data/, models/, the
# HuggingFace cache, Ollama, LM Studio, and ~/models. Reports what's
# loadable and what the daemon would offer.
#
#   make example_list_models
#   ./examples/example_list_models
#
# Set TOY_MODEL_DIR to add another search path:
#
#   TOY_MODEL_DIR=/srv/models ./examples/example_list_models
#
# This is the library surface that a multi-model HTTP daemon would
# call at startup — see lib/model_index.rb. Users compose it into
# their own apps; the daemon shape is one such app.

require_relative "../lib/model_index"

entries = ModelIndex.scan_sources(ModelIndex.default_sources)
ModelIndex.print_summary(entries)
