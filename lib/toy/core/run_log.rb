# lib/toy/core/run_log.rb — Toy::RunLog: read a runs/<id>/ bundle.
#
# CRuby-side (lib/toy/core/ — NO Spinel constraints here, idiomatic
# Ruby is fine; do NOT require this from Spinel-compiled compute).
# Parses runs/<id>/events.jsonl (the toy/v1 stream every trainer
# emits) so "find my best run" is a 3-liner:
#
#   best = Toy::RunLog.scan("runs").first
#   puts best.run_id
#   puts best.final_loss
#
# Surface:
#   RunLog.open(dir)  -> RunLog (fails loud when events.jsonl is
#                        missing or carries no run_start)
#   RunLog.scan(root) -> Array<RunLog> for every subdir with a
#                        parseable events.jsonl, sorted by final_loss
#                        ascending (best first; runs without a loss
#                        sort last)
#   #config      — the run_start event's fields (Hash; includes
#                  "model" and "config" sub-hashes, run_id, phase, …)
#   #steps       — the step events, in order (Array<Hash>)
#   #loss_curve  — the per-step losses (Array<Float>)
#   #final_loss  — run_end's final_loss, else the last step loss,
#                  else nil (a run that never stepped)
#   #run_id, #dir, #run_end — convenience readers
#
# Malformed JSONL LINES fail loud (never silently skipped — a corrupt
# stream should be seen, not averaged over).

require "json"

module Toy
  class RunLog
    attr_reader :dir, :config, :steps, :run_end

    def self.open(dir)
      new(dir)
    end

    # Scan a runs/ root for run bundles. Subdirs without an
    # events.jsonl are skipped (a runs/ root legitimately holds
    # non-run dirs); a PRESENT but unparseable events.jsonl raises.
    # Sorted by final_loss ascending; runs without a final loss last.
    def self.scan(root)
      raise ArgumentError, "RunLog.scan: no such directory: #{root}" unless Dir.exist?(root)
      logs = Dir.children(root).sort.filter_map do |entry|
        dir = File.join(root, entry)
        next unless File.directory?(dir)
        next unless File.file?(File.join(dir, "events.jsonl"))
        new(dir)
      end
      logs.sort_by { |l| l.final_loss || Float::INFINITY }
    end

    def initialize(dir)
      @dir    = dir
      events  = File.join(dir, "events.jsonl")
      raise ArgumentError, "RunLog: no events.jsonl in #{dir}" unless File.file?(events)

      @config  = nil
      @steps   = []
      @run_end = nil
      File.foreach(events).with_index(1) do |line, lineno|
        next if line.strip.empty?
        begin
          ev = JSON.parse(line)
        rescue JSON::ParserError => e
          raise ArgumentError, "RunLog: #{events}:#{lineno}: malformed JSON line (#{e.message})"
        end
        case ev["kind"]
        when "run_start" then @config  = ev
        when "step"      then @steps  << ev
        when "run_end"   then @run_end = ev
        end
      end
      raise ArgumentError, "RunLog: #{events} has no run_start event" if @config.nil?
    end

    def run_id
      @config["run_id"]
    end

    def loss_curve
      @steps.map { |s| s["loss"] }
    end

    # run_end's final_loss when the run completed; else the last step
    # loss (an interrupted run); else nil (a run that never stepped).
    def final_loss
      return @run_end["final_loss"] if @run_end && @run_end["final_loss"]
      @steps.empty? ? nil : @steps.last["loss"]
    end
  end
end
