# lib/toy/serve/openai/api_json.rb -- hand-rolled int-array JSON parser.
#
# MOVED VERBATIM from tep_demo/openai_api_llama.rb:77-134 (P4 toy serve).
# Both the completions handler and the embeddings handler depend on
# ApiJson.get_int_array. It lives here rather than in `Tep::Json` on
# purpose: vendor/spinel/tep/lib/ is a regenerable snapshot of upstream
# Tep, so durable additions belong in the Tep gem itself, not in the
# snapshot. Toy consumes Tep purely as a build-dep transport.
#
# Spinel hygiene (#16): long-hand parsing, no #{} interpolation, no
# Struct.new, String-concat only. KEEP the patterns verbatim.

# Local helper: parse a JSON value at `start_pos` as an int array.
# Returns Array<Int> (empty on absent / non-array / non-int-element).
module ApiJson
  def self.get_int_array(s, key)
    out_ia = [0]
    out_ia.pop
    pos_ia = Tep::Json.find_value_start(s, key)
    if pos_ia < 0
      return out_ia
    end
    pos_ia = Tep::Json.skip_ws(s, pos_ia)
    if pos_ia >= s.length || s[pos_ia] != "["
      return out_ia
    end
    pos_ia += 1
    while pos_ia < s.length
      pos_ia = Tep::Json.skip_ws(s, pos_ia)
      if pos_ia >= s.length
        return out_ia
      end
      if s[pos_ia] == "]"
        return out_ia
      end
      neg_ia = false
      if s[pos_ia] == "-"
        neg_ia = true
        pos_ia += 1
      end
      acc_ia = 0
      saw_digit_ia = false
      while pos_ia < s.length
        ch_ia = s[pos_ia]
        if ch_ia >= "0" && ch_ia <= "9"
          acc_ia = acc_ia * 10 + (ch_ia.bytes[0] - "0".bytes[0])
          saw_digit_ia = true
          pos_ia += 1
        else
          break
        end
      end
      if !saw_digit_ia
        empty_ia = [0]
        empty_ia.pop
        return empty_ia
      end
      if neg_ia
        out_ia.push(0 - acc_ia)
      else
        out_ia.push(acc_ia)
      end
      pos_ia = Tep::Json.skip_ws(s, pos_ia)
      if pos_ia < s.length && s[pos_ia] == ","
        pos_ia += 1
      elsif pos_ia < s.length && s[pos_ia] == "]"
        return out_ia
      end
    end
    out_ia
  end
end
