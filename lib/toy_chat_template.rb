# Per-arch chat templating: list of (role, content) messages → a
# single String the tokenizer can encode. Mirrors HuggingFace's
# `tokenizer.chat_template` but hardcoded per arch family (no Jinja
# evaluator under Spinel).
#
# Usage:
#   text = ToyChatTemplate.apply("chatml",
#            [["system", "Be brief."], ["user", "Hi"]],
#            true)   # true = append generation prompt
#   ids  = tokenizer.encode(text)
#
# Families supported:
#   chatml   — Qwen 2 / Qwen 2.5 / Qwen 3 / SmolLM2 / OLMoE / Mistral-Instruct-v0.3+
#   llama3   — Llama 3 / Llama 3.1 / Llama 3.2 / Llama 3.3
#   mistral  — Mistral-Instruct-v0.1 / v0.2 (pre-ChatML adoption)
#   gemma2   — Gemma 2 (user / model turns)
#
# Detection convention (see ToyChatTemplate.detect_family below):
#   architecture + tokenizer.ggml.model + presence of specific special
#   tokens in the vocab. Returns "chatml" as the safe default for
#   unknown llama-family GGUFs.
#
# Spinel notes:
#   - messages is Array<Array<String>> where each inner array is
#     [role, content]. Symbols not used (Spinel symbol-as-key fights
#     poly dispatch).
#   - No defaults / kwargs (per landmine #4). add_generation_prompt
#     is a regular positional Boolean.

module ToyChatTemplate
  # Render the 4 supported families. Returns the concatenated string;
  # tokenization is the caller's job (existing Tokenizer.encode path).
  def self.apply(family, messages, add_generation_prompt)
    if family == "chatml"
      render_chatml(messages, add_generation_prompt)
    elsif family == "llama3"
      render_llama3(messages, add_generation_prompt)
    elsif family == "mistral"
      render_mistral(messages, add_generation_prompt)
    elsif family == "gemma2"
      render_gemma2(messages, add_generation_prompt)
    else
      # Default to chatml — the broadest covering format among toy's
      # supported arches. Caller wanting strict matching should pass
      # the explicit family string.
      render_chatml(messages, add_generation_prompt)
    end
  end

  # ChatML: <|im_start|>role\ncontent<|im_end|>\n
  # Used by Qwen 2 / Qwen 2.5 / Qwen 3, SmolLM2, OLMoE, modern Mistral.
  def self.render_chatml(messages, add_generation_prompt)
    s = ""
    i = 0
    while i < messages.length
      role    = messages[i][0]
      content = messages[i][1]
      s = s + "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
      i = i + 1
    end
    if add_generation_prompt
      s = s + "<|im_start|>assistant\n"
    end
    s
  end

  # Llama-3 turn structure:
  #   <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n
  #   content<|eot_id|>
  # The system tokenizer auto-prepends <|begin_of_text|> when add_bos
  # is set; we emit it explicitly so the template is self-contained.
  def self.render_llama3(messages, add_generation_prompt)
    s = "<|begin_of_text|>"
    i = 0
    while i < messages.length
      role    = messages[i][0]
      content = messages[i][1]
      s = s + "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
      s = s + content + "<|eot_id|>"
      i = i + 1
    end
    if add_generation_prompt
      s = s + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    end
    s
  end

  # Mistral pre-ChatML: [INST] user [/INST] assistant</s>
  # Subtleties (HF v0.1/v0.2):
  #   - system message gets fused into the FIRST user turn
  #   - [INST]/[/INST] wraps user; assistant text follows verbatim
  #   - </s> closes each assistant turn; final has just [INST] user [/INST]
  def self.render_mistral(messages, add_generation_prompt)
    s = "<s>"
    pending_system = ""
    i = 0
    while i < messages.length
      role    = messages[i][0]
      content = messages[i][1]
      if role == "system"
        # Fuse into next user turn — Mistral v0.x has no system role.
        pending_system = content + "\n\n"
      elsif role == "user"
        s = s + "[INST] " + pending_system + content + " [/INST]"
        pending_system = ""
      elsif role == "assistant"
        s = s + " " + content + "</s>"
      end
      i = i + 1
    end
    # Mistral doesn't need an explicit assistant marker — generation
    # starts right after the last [/INST]. add_generation_prompt is
    # effectively the absence-of-trailing-eos state.
    s
  end

  # Gemma 2 turn structure:
  #   <start_of_turn>role\ncontent<end_of_turn>\n
  # Roles are "user" and "model" (not "assistant"); system messages
  # have no dedicated role — they're fused into the first user turn,
  # like Mistral.
  def self.render_gemma2(messages, add_generation_prompt)
    s = ""
    pending_system = ""
    i = 0
    while i < messages.length
      role    = messages[i][0]
      content = messages[i][1]
      if role == "system"
        pending_system = content + "\n\n"
      elsif role == "user"
        body = pending_system + content
        pending_system = ""
        s = s + "<start_of_turn>user\n" + body + "<end_of_turn>\n"
      elsif role == "assistant" || role == "model"
        s = s + "<start_of_turn>model\n" + content + "<end_of_turn>\n"
      end
      i = i + 1
    end
    if add_generation_prompt
      s = s + "<start_of_turn>model\n"
    end
    s
  end

  # Heuristic family detection from GGUF metadata. The caller passes
  # the arch string from general.architecture plus the tokenizer model
  # kind (tokenizer.ggml.model). Returns a family string suitable for
  # apply().
  #
  # Rules:
  #   - arch=gemma2 → gemma2
  #   - arch=llama AND tokenizer.ggml.tokens contains "<|im_start|>" → chatml
  #     (SmolLM2 + modern Mistral fall here)
  #   - arch=llama AND tokens contains "<|begin_of_text|>" → llama3
  #   - arch=llama (otherwise, classical) → mistral
  #   - arch=qwen2 / qwen3 → chatml
  #   - other → chatml as the modern default
  #
  # `has_im_start` and `has_bot` come from a vocab scan the caller
  # does (tokenizer's @vocab_inv.has_key? on the marker strings).
  def self.detect_family(arch, has_im_start, has_bot)
    if arch == "gemma2"
      "gemma2"
    elsif arch == "qwen2" || arch == "qwen3"
      "chatml"
    elsif arch == "llama"
      if has_im_start
        "chatml"
      elsif has_bot
        "llama3"
      else
        "mistral"
      end
    else
      "chatml"
    end
  end
end
