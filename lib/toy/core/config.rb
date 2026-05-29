# lib/toy/core/config.rb — toy.yml loader.
#
# CRuby ONLY. Minimal project config with exactly TWO keys; an empty
# (or absent-key) toy.yml is valid and yields ALL DEFAULTS — the roadmap
# gate. Unknown keys warn-to-stderr (fail-loud-lite, memory
# feedback_never_mask_fail_loud) but never abort.
#
# Backend is DELIBERATELY NOT a key (design: runtime-autodetected for
# parity). Resist adding model/backend keys.

require "yaml"

module Toy
  module Core
    class Config
      DEFAULTS = {
        # Template for runs/<run_id>/ dir names. Brace-placeholder tokens:
        #   {arch} {date}(YYYYMMDD) {time}(HHMMSS) {seq}(zero-padded daily)
        # Slice 1 stores/echoes only; consumed at train time (P4).
        "run_id_template" => "{arch}-{date}-{seq}",
        # Relative dir the registry/discovery walks for user algos (L1-L4).
        # Slice 1 stores only; discovery is later.
        "algos_path" => "algos"
      }.freeze

      KNOWN_KEYS = DEFAULTS.keys.freeze

      attr_reader :run_id_template, :algos_path, :source_path

      def initialize(values = {})
        merged = DEFAULTS.merge(values)
        @run_id_template = merged["run_id_template"]
        @algos_path = merged["algos_path"]
      end

      # Load from a toy.yml at `dir` (default: cwd). Missing file → all
      # defaults. Empty file → all defaults. Unknown keys → warn.
      def self.load(dir = Dir.pwd, warn_io: $stderr)
        path = File.join(dir, "toy.yml")
        unless File.file?(path)
          c = new
          c.instance_variable_set(:@source_path, nil)
          return c
        end
        raw = YAML.safe_load(File.read(path)) || {}
        unless raw.is_a?(Hash)
          warn_io.puts "toy: warning: toy.yml is not a mapping; ignoring its contents"
          raw = {}
        end
        raw.each_key do |k|
          next if KNOWN_KEYS.include?(k)
          warn_io.puts "toy: warning: unknown toy.yml key #{k.inspect} (ignored)"
        end
        c = new(raw.select { |k, _| KNOWN_KEYS.include?(k) })
        c.instance_variable_set(:@source_path, path)
        c
      end

      def to_h
        { "run_id_template" => @run_id_template, "algos_path" => @algos_path }
      end
    end
  end
end
